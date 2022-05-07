import gc
gc.disable()

import random
import json
import sys
import time
import datetime
import os
from ctypes import cdll
import ctypes
import pickle
import logging
default_logger = logging.getLogger('tunnel.logger')
default_logger.setLevel(logging.CRITICAL)
default_logger.disabled = False

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import io
import sys
import modelnet
import datasource
import medmnist
from medmnist import Evaluator

FUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)

#to server
OP_RECV                      = 0x00
#OP_CLIENT_WAKE_UP            = 0x01
OP_CLIENT_READY              = 0x02
OP_CLIENT_UPDATE             = 0x03
OP_CLIENT_EVAL               = 0x04
#to client
OP_INIT                      = 0x05
OP_REQUEST_UPDATE            = 0x06
OP_STOP_AND_EVAL             = 0x07

def obj_to_pickle_string(x):
    import base64
    return base64.b64encode(pickle.dumps(x))

def pickle_string_to_obj(s):
    import base64
    return pickle.loads(base64.b64decode(s, '-_'))

class GlobalModel(object):
    """docstring for GlobalModel"""
    def __init__(self):
        self.model = self.build_model()
        self.current_weights = self.model.state_dict()

        # for convergence check
        self.prev_train_loss = None

        self.training_start_time = int(round(time.time()))
    
    def build_model(self):
        raise NotImplementedError()

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, new_weights):
        self.current_weights = torch.load(new_weights)
        self.model.load_state_dict(self.current_weights)
       
class GlobalModel_Net(GlobalModel):
    def __init__(self):
        super(GlobalModel_Net, self).__init__()

    def build_model(self):
        data = datasource.MedMNIST()
        model = modelnet.Net(in_channels=data.n_channels, num_classes=data.n_classes)

        return model
       
# Federated Averaging algorithm with the server pulling from clients

class FLServer(object):

    def __init__(self, global_model, host, port, bootaddr):
        self.global_model = global_model()

        self.host = host
        self.port = port
        import uuid
        self.model_id = str(uuid.uuid4())

        #####
        # training states
        self.current_round = -1  # -1 for not yet started
        self.current_round_client_updates = None
        self.eval_client_updates = []
        #####
      
        self.starttime = 0
        self.endtime = 0

        self.lib = cdll.LoadLibrary('./libp2p_peer.so')
        self.lib.Fedcomp_GR.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_byte]
        self.lib.Init_p2p.restype = ctypes.c_char_p

        self.register_handles()

        self.lib.Init_p2p(self.host.encode('utf-8'),int(self.port), int(1), bootaddr.encode('utf-8'))
        if not bootaddr == "" :
            self.lib.Bootstrapping(bootaddr.encode('utf-8'))


    def register_handles(self):
 
        def on_req_global_model(data):
            print("on request global model\n")
            # TODO : need to store on buf directly.
            # error when storing directly on buf
            dic = self.global_model.get_weights()

            #TODO : need to fix
            #buf = io.BytesIO()
            #torch.save(dic, buf)
            torch.save(dic, 'current_global.model')
            with open("current_global.model", "rb") as fd:
                buf = io.BytesIO(fd.read())

            metadata = {
                    'model': buf,
                    'model_id': self.model_id,
            }
            sdata = obj_to_pickle_string(metadata)

            self.lib.SendGlobalModel(sdata, sys.getsizeof(sdata))

        def on_client_update_done_publisher(data):
            print('on client_update_done_publisher\n')
            data = pickle_string_to_obj(data)

            self.global_model.set_weights(data['weights'])

            # gather updates from members
            self.current_round_client_updates = data
            self.current_round = data['round_number']

            print("Round "+str(self.current_round)+'\n')
            print("aggregated train acc : "+str(data['train_acc'] )+'\n')
            print("aggregated train loss : "+str(data['train_loss'] )+'\n')
 
            if 'valid_auc' in self.current_round_client_updates:
                print("aggregated valid acc : "+str(data['valid_acc'] )+'\n')
                print("aggregated valid auc : "+str(data['valid_auc'] )+'\n')
 
            self.global_model.prev_train_loss = data['train_loss']

        def on_client_eval_done_publisher(data):
            print ('on client_eval_done_publisher\n')
            data = pickle_string_to_obj(data)
            #filehandle = open("run.log", "a")
            #filehandle.write ('on client_eval' + str(sys.getsizeof(data))+'\n')
            #filehandle.close()

            if self.eval_client_updates is None:
                return

            self.eval_client_updates = [data]

            print("== done ==\n")
            print("\nfinal test_auc : "+str(data['test_auc'])+'\n')
            print("final test_acc : "+str(data['test_acc'])+'\n')
            self.eval_client_updates = None  # special value, forbid evaling again

        global onreqglobalmodel
        onreqglobalmodel = FUNC(on_req_global_model)
        fnname="on_reqglobalmodel"
        self.lib.Register_callback(fnname.encode('utf-8'),onreqglobalmodel)

        global onclientupdatedonepublisher
        onclientupdatedonepublisher = FUNC(on_client_update_done_publisher)
        fnname="on_clientupdatedone_publisher"
        self.lib.Register_callback(fnname.encode('utf-8'),onclientupdatedonepublisher)

        global onclientevaldonepublisher
        onclientevaldonepublisher = FUNC(on_client_eval_done_publisher)
        fnname="on_clientevaldone_publisher"
        self.lib.Register_callback(fnname.encode('utf-8'),onclientevaldonepublisher)
 
if __name__ == '__main__':
    server = FLServer(GlobalModel_Net, sys.argv[1], sys.argv[2], sys.argv[3])
    filehandle = open("run.log", "w")
    filehandle.write("listening on " + str(sys.argv[1]) + ":" + str(sys.argv[2]) + "\n");
    filehandle.close()
    server.lib.Input()
