import gc
gc.disable()
import os
import sys
import datetime
import time
import pickle
import base64
import numpy as np
import json
from ctypes import cdll
import ctypes

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import io
import modelnet
import datasource
import medmnist
from medmnist import Evaluator

client = None
FUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)
FUNC2 = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int)

#TODO : let Publisher set total training rounds
NUM_GLOBAL_ROUNDS = 1
NUM_LOCAL_EPOCHS = 1 # at each local node

#training variables  -------------
lr = 0.001
# ------------------------


#to server
OP_RECV                      = 0x00
#OP_CLIENT_WAKE_UP            = 0x01 #obsolete
OP_CLIENT_READY              = 0x02
OP_CLIENT_UPDATE             = 0x03
OP_CLIENT_EVAL               = 0x04
#to client
OP_INIT                      = 0x05
OP_REQUEST_UPDATE            = 0x06
OP_STOP_AND_EVAL             = 0x07

def obj_to_pickle_string(x):
    return base64.b64encode(pickle.dumps(x))

def pickle_string_to_obj(s):
    return pickle.loads(base64.b64decode(s, '-_'))

class LocalModel(object):
    def __init__(self, model_config, datasource):
        # for convergence check
        self.prev_train_loss = None

        # all rounds; losses[i] = [round#, timestamp, loss]
        # round# could be None if not applicable
        self.train_losses = []
        self.train_accs = []
        self.valid_aucs = []
        self.valid_accs = []

        self.model_config = model_config
        self.model_id = model_config['model_id']

        self.model = modelnet.Net(in_channels=datasource.n_channels, num_classes=datasource.n_classes)
        self.datasource = datasource
        # define loss function and optimizer
        if datasource.task == "multi-label, binary-class":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer =  optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
	
        self.current_weights = None

        self.x_train = len(self.datasource.train_loader.dataset.imgs)
        self.x_valid = len(self.datasource.valid_loader.dataset.imgs)
        self.x_test = len(self.datasource.test_loader.dataset.imgs)

        self.training_start_time = int(round(time.time()))

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, new_weights):
        self.current_weights = torch.load(new_weights)
        self.model.load_state_dict(self.current_weights)

    def train_one_round(self):
        train_losses=[]
        train_accu=[]
        for epoch in range(NUM_LOCAL_EPOCHS):
            train_correct = 0
            train_total = 0
            test_correct = 0
            test_total = 0

            running_loss = 0
            correct = 0
            total = 0

            self.model.train()
            for inputs, targets in tqdm(self.datasource.train_loader):
                # forward + backward + optimize
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                if self.datasource.task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    loss = self.criterion(outputs, targets)
                else:
                    targets = targets.squeeze().long()
                    loss = self.criterion(outputs, targets)

                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            train_loss=running_loss/len(self.datasource.train_loader)
            accu=100.*correct/total

            train_accu.append(accu)
            train_losses.append(train_loss)
            print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))

        return self.model.state_dict(), train_loss, accu

    def test(self, split):
        self.model.eval()
        y_true = torch.tensor([])
        y_score = torch.tensor([])

        data_loader = self.datasource.valid_loader if split == 'valid' else self.datasource.test_loader

        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = self.model(inputs)

                if self.datasource.task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    outputs = outputs.softmax(dim=-1)
                else:
                    targets = targets.squeeze().long()
                    outputs = outputs.softmax(dim=-1)
                    targets = targets.float().resize_(len(targets), 1)

                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)

            y_true = y_true.numpy()
            y_score = y_score.detach().numpy()

            evaluator = Evaluator(self.datasource.data_flag, split)
            metrics = evaluator.evaluate(y_score)

            return split, metrics[0], metrics[1]

    def validate(self):
        print('start validation')
        split, auc, acc = self.test('valid')
        print('%s  auc: %.3f  acc:%.3f' % (split, auc, acc))

        return auc, acc

    def evaluate(self):
        print('start evaluation')
        split, auc, acc = self.test('test')
        print('%s  auc: %.3f  acc:%.3f' % (split, auc, acc))

        return auc, acc

    #TODO : sub-leader aggregation fix
    def update_weights(self, client_weights, client_sizes):
        global_dict = self.model.state_dict()
        total_size = sum(client_sizes)
        n = len(client_weights)
        for k in global_dict.keys():
            global_dict[k] = torch.stack([client_weights[i][k].float()*(n*client_sizes[i]/total_size) for i in range(len(client_weights))], 0).mean(0)
        self.model.load_state_dict(global_dict)
        self.current_weights = global_dict

    def aggregate_auc_acc(self, client_aucs, client_accs, client_sizes):
        total_size = np.sum(client_sizes)
        # weighted sum
        aggr_auc = np.sum(client_aucs[i] / total_size * client_sizes[i]
                for i in range(len(client_sizes)))
        aggr_acc = np.sum(client_accs[i] / total_size * client_sizes[i]
                for i in range(len(client_sizes)))
        return aggr_auc, aggr_acc, total_size

    def aggregate_loss_acc(self, client_losses, client_accs, client_sizes):
        total_size = np.sum(client_sizes)
        # weighted sum
        aggr_loss = np.sum(client_losses[i] / total_size * client_sizes[i]
                for i in range(len(client_sizes)))
        aggr_acc = np.sum(client_accs[i] / total_size * client_sizes[i]
                for i in range(len(client_sizes)))
        return aggr_loss, aggr_acc, total_size

    def aggregate_train_loss_acc(self, client_losses, client_accs, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_acc, aggr_size = self.aggregate_loss_acc(client_losses, client_accs, client_sizes)

        self.train_losses += [[cur_round, cur_time, aggr_loss]]
        self.train_accs += [[cur_round, cur_time, aggr_acc]]
        with open('stats.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_acc

    def aggregate_valid_auc_acc(self, client_aucs, client_accs, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_auc, aggr_acc, aggr_size = self.aggregate_auc_acc(client_aucs, client_accs, client_sizes)
        self.valid_aucs += [[cur_round, cur_time, aggr_auc]]
        self.valid_accs += [[cur_round, cur_time, aggr_acc]]
        with open('stats.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_auc, aggr_acc

    def get_stats(self):
        return {
            "train_loss": self.train_losses,
            "train_acc": self.train_accs,
            "valid_auc": self.valid_aucs,
            "valid_acc": self.valid_accs
        }

class FederatedClient(object):
    MIN_NUM_WORKERS = 0 #total from this branch. This will be set by grouping protocol during grouping
    def __init__(self, host, port, bootaddr):
        self.local_model = None

        # You may want to have IID or non-IID setting based on number of your peers 
        # by default, this code brings all dataset
        self.datasource = datasource.MedMNIST()

        self.current_round = 0
        self.current_round_client_updates = []
        self.eval_client_updates = []

        self.port = int(port)
 
        print("p2p init")
        self.lib = cdll.LoadLibrary('./libp2p_peer.so')
        self.lib.Init_p2p.restype = ctypes.c_char_p
        self.lib.Fedcomp_GR.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_byte]
        self.lib.Report_GR.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_byte, ctypes.c_int]

        self.register_handles()
        self.lib.Init_p2p(host.encode('utf-8'),int(port), int(0), bootaddr.encode('utf-8'))

        self.lib.Bootstrapping(bootaddr.encode('utf-8'))


    def register_handles(self):

        def on_set_num_client(num):
            print('APP : on set_num_client')
            self.MIN_NUM_WORKERS = num
            print('APP : set MIN_NUM_WORKERS ',self.MIN_NUM_WORKERS)

        def on_init_subleader(data):
            print('APP : on init_subleader')
            model_config = pickle_string_to_obj(data)

            self.local_model = LocalModel(model_config, self.datasource)
            self.local_model.set_weights(model_config['model'])

            self.lib.IncreaseNumClientReady()
            
        def on_init_worker(data):
            print('APP : on init_worker')
            model_config = pickle_string_to_obj(data)

            self.local_model = LocalModel(model_config, self.datasource)
            self.local_model.set_weights(model_config['model'])

            print("send client_ready to upper leader\n")
            self.lib.Report_GR(None, 0, OP_CLIENT_READY, 1)

        # handler for initiator role
        def on_global_model(data):
            print('APP : on global model')
            model_config = pickle_string_to_obj(data)

            self.lib.IncreaseNumClientReady()

            self.local_model = LocalModel(model_config, self.datasource)
            self.local_model.set_weights(model_config['model'])

            # TODO : need to fix
            #buf = io.BytesIO()
            #torch.save(self.local_model.current_weights, buf)
            torch.save(self.local_model.current_weights, 'current_local'+str(self.port)+'.model')
            with open('current_local'+str(self.port)+'.model', "rb") as fd:
                buf = io.BytesIO(fd.read())

            # global model dissemination
            metadata = {
                'model': buf,
                'model_id': model_config['model_id']
            }
            sdata = obj_to_pickle_string(metadata)
            self.lib.Fedcomp_GR(sdata, sys.getsizeof(sdata),OP_INIT)

        def on_train_my_model(arg):
            # train my model
            start = datetime.datetime.now()

            self.local_model.current_weights, train_loss, train_acc = self.local_model.train_one_round()

            self.lib.IncreaseNumClientUpdateInitiator()

            end = datetime.datetime.now()

            diff = end - start
            print("diff(sec) : " + str(diff.seconds)+str("\n"))
            self.lib.RecordMyTrainTime(diff.seconds)

        #subleader handler
        def on_client_update_subleader(data):
            print('APP : on client_update_subleader \n')
            data = pickle_string_to_obj(data)

            # gather updates and discard outdated update
            if data['round_number'] == self.current_round:
                data['weights'] = torch.load(data['weights'])
                self.current_round_client_updates += [data]

        #initiator handler
        def on_client_update_initiator(data):
            print('on client_update_initiator\n')
            data = pickle_string_to_obj(data)
            #filehandle = open("run.log", "a")
            #filehandle.write ('on client_update: datasize :' + str(sys.getsizeof(data))+'\n')

            # gather updates from members
            if data['round_number'] == self.current_round:
                data['weights'] = torch.load(data['weights'])
                self.current_round_client_updates += [data]

        def on_client_update_done_initiator(arg):
            print('on client_update_done_initiator\n')
            self.local_model.update_weights(
                [x['weights'] for x in self.current_round_client_updates],
                [x['train_size'] for x in self.current_round_client_updates],
            )
            aggr_train_loss, aggr_train_acc = self.local_model.aggregate_train_loss_acc(
                [x['train_loss'] for x in self.current_round_client_updates],
                [x['train_acc'] for x in self.current_round_client_updates],
                [x['train_size'] for x in self.current_round_client_updates],
                self.current_round
            )
            #filehandle = open("run.log", "a")
            #filehandle.write("aggr_train_loss"+str(aggr_train_loss)+'\n')
            #filehandle.write("aggr_train_acc"+str(aggr_train_acc)+'\n')
            #filehandle.close()

            if 'valid_auc' in self.current_round_client_updates[0]:
                aggr_valid_auc, aggr_valid_acc = self.local_model.aggregate_valid_auc_acc(
                [x['valid_auc'] for x in self.current_round_client_updates],
                [x['valid_acc'] for x in self.current_round_client_updates],
                [x['valid_size'] for x in self.current_round_client_updates],
                self.current_round
                )
                #filehandle = open("run.log", "a")
                #filehandle.write("aggr_valid_aucs"+str(aggr_valid_aucs)+'\n')
                #filehandle.write("aggr_valid_acc"+str(aggr_valid_acc)+'\n')
                #filehandle.close()

	    #TODO : this comment is for test. remove later. we need to stop when it converges.
            #if self.local_model.prev_train_loss is not None and \
            #        (self.local_model.prev_train_loss - aggr_train_loss) / self.local_model.prev_train_loss < .01:
            #    # converges
            #    filehandle = open("run.log", "a")
            #    filehandle.write("converges! starting test phase..")
            #    filehandle.close()
            #    self.stop_and_eval()
            #    return
            #self.local_model.prev_train_loss = aggr_train_loss

            # TODO : need to fix
            #buf = io.BytesIO()
            #torch.save(self.local_model.current_weights, buf)
            torch.save(self.local_model.current_weights, 'current_local'+str(self.port)+'.model')
            with open('current_local'+str(self.port)+'.model', "rb") as fd:
                buf = io.BytesIO(fd.read())

            if self.current_round >= NUM_GLOBAL_ROUNDS:
                # report to publisher. send the aggregated weight
                resp = {
                    'round_number': self.current_round,
                    'weights': buf,
                    'train_size': self.local_model.x_train,
                    'valid_size': self.local_model.x_valid,
                    'train_loss': aggr_train_loss,
                    'train_acc': aggr_train_acc,
                }

                sresp = obj_to_pickle_string(resp)
                print('send CLIENT_UPDATE to publisher, msg payload size:' + str(sys.getsizeof(sresp)) + '\n' )
                self.lib.Report_GR(sresp, sys.getsizeof(sresp), OP_CLIENT_UPDATE, 0)

                # send stop and eval request to members
                self.stop_and_eval()
                # eval my model
                test_auc, test_acc = self.local_model.evaluate()
                resp = {
                    'test_size': self.local_model.x_test,
                    'test_auc': test_auc,
                    'test_acc': test_acc
                }
                self.eval_client_updates += [resp]
                self.lib.IncreaseNumClientEvalInitiator()
            else:
                # report to publisher. send the aggregated weight
                resp = {
                    'round_number': self.current_round,
                    'weights': buf,
                    'train_size': self.local_model.x_train,
                    'valid_size': self.local_model.x_valid,
                    'train_loss': aggr_train_loss,
                    'train_acc': aggr_train_acc,
                }
                sresp = obj_to_pickle_string(resp)
                print('send CLIENT_UPDATE to publisher, msg payload size:' + str(sys.getsizeof(sresp)) + '\n' )
                self.lib.Report_GR(sresp, sys.getsizeof(sresp), OP_CLIENT_UPDATE, 0)

                # send request updates to the members
                self.train_next_round()

                start = datetime.datetime.now()

                # train my model
                self.local_model.current_weights, train_loss, train_acc = self.local_model.train_one_round()

                # increase update done counter
                self.lib.IncreaseNumClientUpdateInitiator()

                end = datetime.datetime.now()
                diff = end - start
                print("diff(sec) : " + str(diff.seconds)+str("\n"))
                self.lib.RecordMyTrainTime(diff.seconds)

        # subleader handler
        def on_request_update_subleader(data):
            data = pickle_string_to_obj(data)
            print('APP : on request_update \n')

            self.current_round_client_updates = []

            self.current_round = data['round_number']
            print("round_number : "+str(data['round_number'])+"\n")

            #filehandle = open("run.log", "a")
            #filehandle.write ('on request_update received data size :' +str(sys.getsizeof(args)) + '\n')
            start = datetime.datetime.now()
            #filehandle.writelines("start : " + str(start)+str("\n"))
            #filehandle.close()

            # train my model
            self.local_model.set_weights(data['current_weights'])
            self.local_model.current_weights, train_loss, train_acc = self.local_model.train_one_round()

            self.lib.IncreaseNumClientUpdate()

            end = datetime.datetime.now()
            diff = end - start
            print("diff(sec) : " + str(diff.seconds)+str("\n"))
            self.lib.RecordMyTrainTime(diff.seconds)

            resp = {
                'round_number': data['round_number'],
                'weights': self.local_model.model.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_size': self.local_model.x_train,
            }
            #filehandle = open("run.log", "a")
            #filehandle.write ('train_loss' + str(train_loss) + '\n' )
            #filehandle.write ('train_acc' + str(train_acc) + '\n' )

            self.current_round_client_updates += [resp]

        # worker handler
        def on_request_update_worker(data):
            print('APP : on request_update_worker\n')
            data = pickle_string_to_obj(data)

            self.current_round = data['round_number']
            print("round_number : "+str(data['round_number'])+"\n")

            #filehandle = open("run.log", "a")
            #filehandle.write ('on request_update received data size :' +str(sys.getsizeof(args)) + '\n')
            #start = datetime.datetime.now()
            #filehandle.writelines("start : " + str(start)+str("\n"))
            #filehandle.close()

            start = datetime.datetime.now()

            self.local_model.set_weights(data['current_weights'])
            self.local_model.current_weights, train_loss, train_acc = self.local_model.train_one_round()

            end = datetime.datetime.now()

            diff = end - start
            print("diff(sec) : " + str(diff.seconds)+str("\n"))
            self.lib.RecordMyTrainTime(diff.seconds)

            #filehandle = open("run.log", "a")
            #filehandle.writelines("end : " + str(end)+str("\n"))
            #filehandle.writelines("diff(s) : " + str(diff.seconds)+str("\n"))
            #filehandle.writelines("diff(us) : " + str(diff.microseconds)+str("\n"))
            #filehandle.close()

            # TODO : need to fix
            #buf = io.BytesIO()
            #torch.save(self.local_model.current_weights, buf)
            torch.save(self.local_model.current_weights, 'current_local'+str(self.port)+'.model')
            with open('current_local'+str(self.port)+'.model', "rb") as fd:
                buf = io.BytesIO(fd.read())

            resp = {
                'round_number': data['round_number'],
                'weights': buf,
                'train_size': self.local_model.x_train,
                'valid_size': self.local_model.x_valid,
                'train_loss': train_loss,
                'train_acc': train_acc,
            }
            #filehandle = open("run.log", "a")
            #filehandle.write ('train_loss' + str(train_loss) + '\n' )
            #filehandle.write ('train_acc' + str(train_acc) + '\n' )

            #print('start validate')
            #valid_auc, valid_acc = self.local_model.validate()
            #resp['valid_auc'] = valid_auc
            #resp['valid_acc'] = valid_acc

            #filehandle.write ('valid_auc' + str(valid_auc) + '\n' )
            #filehandle.write ('valid_acc' + str(valid_acc) + '\n' )
            #filehandle.close()

            sresp = obj_to_pickle_string(resp)
            print('send CLIENT_UPDATE to upper leader train_size:' +str(resp['train_size']) + '\n' )
            self.lib.Report_GR(sresp, sys.getsizeof(sresp), OP_CLIENT_UPDATE, 1)

        # sub-leader handler
        def on_stop_and_eval_subleader(data):
            data = pickle_string_to_obj(data)
            print('APP : on stop_and_eval_subleader')
            #filehandle = open("run.log", "a")
            #filehandle.write ('on stop_and_eval received data size :' +str(sys.getsizeof(args)) + '\n')

            #filehandle.write ('send CLIENT_EVAL to size:' + str(sys.getsizeof(sresp)) + '\n' )
            #filehandle.close()

            self.local_model.set_weights(data['current_weights'])
            test_auc, test_acc = self.local_model.evaluate()

            resp = {
                'test_size': self.local_model.x_test,
                'test_auc': test_auc,
                'test_acc': test_acc
            }

            self.eval_client_updates += [resp]

            self.lib.IncreaseNumClientEval()

        # worker handler
        def on_stop_and_eval_worker(data):
            print('APP : on stop_and_eval')
            data = pickle_string_to_obj(data)
            #filehandle = open("run.log", "a")
            #filehandle.write ('on stop_and_eval received data size :' +str(sys.getsizeof(args)) + '\n')

            self.local_model.set_weights(data['current_weights'])
            test_auc, test_acc = self.local_model.evaluate()
            resp = {
                'test_size': self.local_model.x_test,
                'test_auc': test_auc,
                'test_acc': test_acc
            }
            #filehandle.write ('send CLIENT_EVAL size:' + str(sys.getsizeof(sresp)) + '\n' )
            #filehandle.close()
            sdata = obj_to_pickle_string(resp)
            print('APP : on stop_and_eval: report')
            self.lib.Report_GR(sdata, sys.getsizeof(sdata), OP_CLIENT_EVAL, 1)

        def on_client_eval_subleader(data):
            data = pickle_string_to_obj(data)
            print ('APP : on client_eval_subleader\n')

            if self.eval_client_updates is None:
                return

            self.eval_client_updates += [data]

        #initiator handler
        def on_client_eval_initiator(data):
            data = pickle_string_to_obj(data)
            print ('APP : on client_eval\n')

            if self.eval_client_updates is None:
                return

            self.eval_client_updates += [data]

        def on_client_eval_done_initiator(arg):
            aggr_test_auc, aggr_test_acc, aggr_test_size = self.local_model.aggregate_auc_acc(
            [x['test_auc'] for x in self.eval_client_updates],
            [x['test_acc'] for x in self.eval_client_updates],
            [x['test_size'] for x in self.eval_client_updates],
            );
            filehandle = open("run.log", "a")
            filehandle.write("\nfinal aggr_test_auc : "+str(aggr_test_auc)+'\n')
            filehandle.write("final aggr_test_acc : "+str(aggr_test_acc)+'\n')
            filehandle.write("== done ==\n")
            print("== done ==\n")
            print("\nfinal aggr_test_auc : "+str(aggr_test_auc)+'\n')
            print("final aggr_test_acc : "+str(aggr_test_acc)+'\n')
            #self.end = int(round(time.time()))
            #filehandle.write("end : " + str(self.end)+'\n')
            #print("end : " + str(self.end)+'\n')
            #filehandle.write("diff : " + str(self.end - self.start)+'\n')
            #print("diff : " + str(self.end - self.start)+'\n')
            #filehandle.write("== done ==\n")
            #filehandle.close()
            #self.eval_client_updates = None  # special value, forbid evaling again

            #report to publisher
            resp = {
                'test_size': aggr_test_size,
                'test_auc': aggr_test_auc,
                'test_acc': aggr_test_acc
            }
            sdata = obj_to_pickle_string(resp)
            self.lib.Report_GR(sdata, sys.getsizeof(sdata), OP_CLIENT_EVAL, 0)

        def on_report_client_update(aggregation_num):
            print( "APP : report client update\n") 
            print(len(self.current_round_client_updates))
            print(self.current_round_client_updates[-1]['train_size'])
            self.local_model.update_weights(
                [x['weights'] for x in self.current_round_client_updates],
                [x['train_size'] for x in self.current_round_client_updates],
            )
            aggr_train_loss, aggr_train_acc = self.local_model.aggregate_train_loss_acc(
                [x['train_loss'] for x in self.current_round_client_updates],
                [x['train_acc'] for x in self.current_round_client_updates],
                [x['train_size'] for x in self.current_round_client_updates],
                self.current_round
            )

            # TODO : need to fix
            #buf = io.BytesIO()
            #torch.save(self.local_model.current_weights, buf)
            torch.save(self.local_model.current_weights, 'current_local'+str(self.port)+'.model')
            with open('current_local'+str(self.port)+'.model', "rb") as fd:
                buf = io.BytesIO(fd.read())

            resp = {
                'round_number': self.current_round,
                'weights': buf,
                'train_size': self.local_model.x_train,
                'valid_size': self.local_model.x_valid,
                'train_loss': aggr_train_loss,
                'train_acc': aggr_train_acc,
            }

            if 'valid_auc' in self.current_round_client_updates[0]:
                aggr_valid_auc, aggr_valid_acc = self.local_model.aggregate_valid_auc_acc(
                [x['valid_auc'] for x in self.current_round_client_updates],
                [x['valid_acc'] for x in self.current_round_client_updates],
                [x['valid_size'] for x in self.current_round_client_updates],
                self.current_round
                )
                resp['valid_auc'] = aggr_valid_auc
                resp['valid_acc'] = aggr_valid_acc
 
            sresp = obj_to_pickle_string(resp)
            print('send CLIENT_UPDATE to server, msg payload size:' + str(sys.getsizeof(sresp)) + '\n' )
            self.lib.Report_GR(sresp, sys.getsizeof(sresp), OP_CLIENT_UPDATE, aggregation_num)

        def on_train_next_round(arg):
            self.current_round += 1
            # buffers all client updates
            self.current_round_client_updates = []

            #filehandle = open("run.log", "a")
            #filehandle.write("### Round "+str(self.current_round)+"###\n")
            print("### Round "+str(self.current_round)+"###\n")
            #filehandle.close()

            # TODO : need to fix
            #buf = io.BytesIO()
            #torch.save(self.local_model.current_weights, buf)
            torch.save(self.local_model.current_weights, 'current_local'+str(self.port)+'.model')
            with open('current_local'+str(self.port)+'.model', "rb") as fd:
                buf = io.BytesIO(fd.read())

            metadata = {
                'model_id': self.local_model.model_id,
                'round_number': self.current_round,
                'current_weights': buf
            }
            sdata = obj_to_pickle_string(metadata)
            self.lib.Fedcomp_GR(sdata, sys.getsizeof(sdata), OP_REQUEST_UPDATE)
            print("request_update sent\n")

        def on_report_client_eval(aggregation_num):
            aggr_test_auc, aggr_test_acc, aggr_test_size = self.local_model.aggregate_auc_acc(
                [x['test_auc'] for x in self.eval_client_updates],
                [x['test_acc'] for x in self.eval_client_updates],
                [x['test_size'] for x in self.eval_client_updates],
            );
            self.eval_client_updates = None  # special value, forbid evaling again
            resp = {
                'test_size': aggr_test_size,
                'test_auc': aggr_test_auc,
                'test_acc': aggr_test_acc
            }
            #filehandle.write ('send CLIENT_EVAL size:' + str(sys.getsizeof(sresp)) + '\n' )
            #filehandle.close()
            sdata = obj_to_pickle_string(resp)
            self.lib.Report_GR(sdata, sys.getsizeof(sdata), OP_CLIENT_EVAL, aggregation_num)

        global onsetnumclient
        onsetnumclient = FUNC2(on_set_num_client)
        fnname="on_set_num_client"
        self.lib.Register_callback(fnname.encode('utf-8'),onsetnumclient)

        global onglobalmodel
        onglobalmodel = FUNC(on_global_model)
        fnname="on_global_model"
        self.lib.Register_callback(fnname.encode('utf-8'),onglobalmodel)

        global oninitworker
        oninitworker = FUNC(on_init_worker)
        fnname="on_init_worker"
        self.lib.Register_callback(fnname.encode('utf-8'),oninitworker)

        global oninitsubleader
        oninitsubleader = FUNC(on_init_subleader)
        fnname="on_init_subleader"
        self.lib.Register_callback(fnname.encode('utf-8'),oninitsubleader)

        global onrequestupdateworker
        onrequestupdateworker = FUNC(on_request_update_worker)
        fnname="on_request_update_worker"
        self.lib.Register_callback(fnname.encode('utf-8'),onrequestupdateworker)

        global onrequestupdatesubleader
        onrequestupdatesubleader = FUNC(on_request_update_subleader)
        fnname="on_request_update_subleader"
        self.lib.Register_callback(fnname.encode('utf-8'),onrequestupdatesubleader)

        global onstopandevalworker
        onstopandevalworker = FUNC(on_stop_and_eval_worker)
        fnname="on_stop_and_eval_worker"
        self.lib.Register_callback(fnname.encode('utf-8'),onstopandevalworker)

        global onstopandevalsubleader
        onstopandevalsubleader = FUNC(on_stop_and_eval_subleader)
        fnname="on_stop_and_eval_subleader"
        self.lib.Register_callback(fnname.encode('utf-8'),onstopandevalsubleader)

        global onclientupdatesubleader
        onclientupdatesubleader = FUNC(on_client_update_subleader)
        fnname="on_clientupdate_subleader"
        self.lib.Register_callback(fnname.encode('utf-8'),onclientupdatesubleader)

        global onclientupdateinitiator
        onclientupdateinitiator = FUNC(on_client_update_initiator)
        fnname="on_clientupdate_initiator"
        self.lib.Register_callback(fnname.encode('utf-8'),onclientupdateinitiator)

        global onclientupdatedoneinitiator
        onclientupdatedoneinitiator = FUNC(on_client_update_done_initiator)
        fnname="on_clientupdatedone_initiator"
        self.lib.Register_callback(fnname.encode('utf-8'),onclientupdatedoneinitiator)

        global onclientevalsubleader
        onclientevalsubleader = FUNC(on_client_eval_subleader)
        fnname="on_clienteval_subleader"
        self.lib.Register_callback(fnname.encode('utf-8'),onclientevalsubleader)

        global onclientevalinitiator
        onclientevalinitiator = FUNC(on_client_eval_initiator)
        fnname="on_clienteval_initiator"
        self.lib.Register_callback(fnname.encode('utf-8'),onclientevalinitiator)

        global onclientevaldoneinitiator
        onclientevaldoneinitiator = FUNC(on_client_eval_done_initiator)
        fnname="on_clientevaldone_initiator"
        self.lib.Register_callback(fnname.encode('utf-8'),onclientevaldoneinitiator)

        global onreportclientupdate
        onreportclientupdate = FUNC2(on_report_client_update)
        fnname="on_report_client_update"
        self.lib.Register_callback(fnname.encode('utf-8'),onreportclientupdate)

        global ontrainnextround
        ontrainnextround = FUNC(on_train_next_round)
        fnname="on_train_next_round"
        self.lib.Register_callback(fnname.encode('utf-8'),ontrainnextround)

        global onreportclienteval
        onreportclienteval = FUNC2(on_report_client_eval)
        fnname="on_report_client_eval"
        self.lib.Register_callback(fnname.encode('utf-8'),onreportclienteval)

        global ontrainmymodel
        ontrainmymodel = FUNC(on_train_my_model)
        fnname="on_train_my_model"
        self.lib.Register_callback(fnname.encode('utf-8'),ontrainmymodel)


    #internal function
    # Note: we assume that during training the #workers will be >= MIN_NUM_WORKERS
    def train_next_round(self):
        self.current_round += 1
        # buffers all client updates
        self.current_round_client_updates = []

        #filehandle = open("run.log", "a")
        #filehandle.write("### Round "+str(self.current_ro(und)+"###\n")
        print("### Round "+str(self.current_round)+"###\n")
        #filehandle.close()

        # TODO : need to fix
        #buf = io.BytesIO()
        #torch.save(self.local_model.current_weights, buf)
        torch.save(self.local_model.current_weights, 'current_local'+str(self.port)+'.model')
        with open('current_local'+str(self.port)+'.model', "rb") as fd:
            buf = io.BytesIO(fd.read())

        metadata = {
            'model_id': self.local_model.model_id,
            'round_number': self.current_round,
            'current_weights': buf,
        }
        sdata = obj_to_pickle_string(metadata)
        self.lib.Fedcomp_GR(sdata, sys.getsizeof(sdata), OP_REQUEST_UPDATE)
        print("request_update sent\n")

    def stop_and_eval(self):
        self.eval_client_updates = []

        # TODO : need to fix
        #buf = io.BytesIO()
        #torch.save(self.local_model.current_weights, buf)
        torch.save(self.local_model.current_weights, 'current_local'+str(self.port)+'.model')
        with open('current_local'+str(self.port)+'.model', "rb") as fd:
            buf = io.BytesIO(fd.read())

        metadata = {
            'model_id': self.local_model.model_id,
            'current_weights': buf
        }
        sdata = obj_to_pickle_string(metadata)
        self.lib.Fedcomp_GR(sdata, sys.getsizeof(sdata), OP_STOP_AND_EVAL)

#global client
if __name__ == "__main__":
    filehandle = open("run.log", "w")
    filehandle.write("running client \n")
    filehandle.close()

    client = FederatedClient(sys.argv[1], sys.argv[2], sys.argv[3])

    # If you use run.sh to launch many peer nodes, you should comment below line
    #client.lib.Input()
    # Instead use this to block the process
    while True:
        pass
    
