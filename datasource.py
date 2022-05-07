import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO

download = True

BATCH_SIZE = 128


class DataSource(object):
    def __init__(self):
        raise NotImplementedError()
    def partitioned_by_rows(self, num_workers, test_reserve=.3):
        raise NotImplementedError()
    def sample_single_non_iid(self, weight=None):
        raise NotImplementedError()

# You may want to have IID or non-IID setting based on number of your peers 
# by default, this code brings all dataset
class MedMNIST(DataSource):

    def __init__(self):
        self.data_flag = 'pathmnist' 
        info = INFO[self.data_flag]
        self.n_channels = info['n_channels']
        self.n_classes = len(info['label'])
        self.task = info['task']

        DataClass = getattr(medmnist, info['python_class'])

        # preprocessing
        data_transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[.5], std=[.5])
	])

        # load the data
        train_dataset = DataClass(split='train', transform=data_transform, download=download)
        test_dataset = DataClass(split='test', transform=data_transform, download=download)

        self.pil_dataset = DataClass(split='train', download=download)

        # encapsulate data into dataloader form
        self.train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.valid_loader = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
        self.test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

        print(train_dataset)
        print("===================")
        print(test_dataset)

if __name__ == "__main__":
    m = MedMNIST()

