import torch.utils.data
from dataloader.base_data_loader import BaseDataLoader
from dataloader.tif_dataset import MultiBandTifDataset
from torch.utils.data import random_split
class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'
    
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        
        self.dataset = MultiBandTifDataset()
        print("dataset [%s] was created" % (self.dataset.name()))
        self.dataset.initialize(opt)

        val_ratio = getattr(opt, "val_ratio", 0.0)  
        if not (0.0 <= val_ratio < 1.0):
            raise ValueError("val_ratio must be in [0, 1). Got %s" % val_ratio)
        if val_ratio > 0.0:
            total_len = len(self.dataset)
            val_len = int(total_len * val_ratio)
            train_len = total_len - val_len
            g = torch.Generator().manual_seed(getattr(opt, "seed", 42))
            train_set, val_set = random_split(self.dataset, [train_len, val_len], generator=g)
        else:
            train_set, val_set = self.dataset, None

        self.dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))
        
        self.val_dataloader = None
        if val_set is not None:
            self.val_dataloader = torch.utils.data.DataLoader(
                val_set,
                batch_size=opt.batchSize,
                shuffle=False,
                num_workers=int(opt.nThreads))
    
    def load_data(self):
        return self.dataloader

    def load_val_data(self):
        return self.val_dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader