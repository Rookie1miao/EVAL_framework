import os.path
import torch
import torch.utils.data
import numpy as np
import rasterio
from dataloader.base_dataset import BaseDataset, get_params
from dataloader.base_data_loader import BaseDataLoader
from dataloader.image_folder import make_dataset

class MultiBandTifDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        dir_A = '_feature'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted([p for p in make_dataset(self.dir_A) if p.endswith('.tif')])
        dir_D = '_dmsp'
        self.dir_D = os.path.join(opt.dataroot, opt.phase + dir_D)
        self.D_paths = sorted([p for p in make_dataset(self.dir_D) if p.endswith('.tif')])
        dir_I = '_instance'
        self.dir_I = os.path.join(opt.dataroot, opt.phase + dir_I)
        self.I_paths = sorted([p for p in make_dataset(self.dir_I) if p.endswith('.tif')])

        if opt.isTrain:
            dir_B = '_target'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
            self.B_paths = sorted([p for p in make_dataset(self.dir_B) if p.endswith('.tif')])

        if len(self.A_paths) == 0:
            raise ValueError(f"cannot find any files in {self.dir_A}")
        if opt.isTrain and len(self.B_paths) == 0:
            raise ValueError(f"cannot find any files in {self.dir_B}")
        if len(self.I_paths) == 0:
            raise ValueError(f"cannot find any files in {self.dir_I}")

        self.dataset_size = len(self.A_paths)
        
        self.num_bands_feature = 6
        self.num_bands_dmsp = 1
        self.num_bands_target = opt.output_nc if hasattr(opt, 'output_nc') else 1
        self.num_bands_instance = 1


    def __getitem__(self, index):
        A_path = self.A_paths[index]
        with rasterio.open(A_path) as src:
            available_bands = src.count
            if available_bands < self.num_bands_feature:
                raise ValueError(f"文件 {A_path} 的波段数({available_bands})小于要求的波段数({self.num_bands_feature})")
            
            A_array = src.read([i for i in range(1, self.num_bands_feature + 1)])

        A_tensor = torch.from_numpy(A_array).float()

        D_path = self.D_paths[index]
        with rasterio.open(D_path) as src:
            available_bands = src.count
            if available_bands < self.num_bands_dmsp:
                raise ValueError(f"文件 {D_path} 的波段数({available_bands})小于要求的波段数({self.num_bands_dmsp})")
            
            D_array = src.read([i for i in range(1, self.num_bands_dmsp + 1)])
            
        D_tensor = torch.from_numpy(D_array).float()

        B_tensor = torch.tensor(0)
        
        I_tensor = torch.tensor(0)

        if self.opt.isTrain:
            B_path = self.B_paths[index]
            with rasterio.open(B_path) as src:
                available_bands = src.count
                if available_bands < self.num_bands_target:
                    raise ValueError(f"文件 {B_path} 的波段数({available_bands})小于要求的波段数({self.num_bands_target})")
                
                B_array = src.read([i for i in range(1, self.num_bands_target + 1)])

            B_tensor = torch.from_numpy(B_array).float()

        I_path = self.I_paths[index]
        with rasterio.open(I_path) as src:
            available_bands = src.count
            if available_bands < self.num_bands_instance:
                raise ValueError(f"文件 {I_path} 的波段数({available_bands})小于要求的波段数({self.num_bands_instance})")
                
            I_array = src.read([i for i in range(1, self.num_bands_instance + 1)])

        I_tensor = torch.from_numpy(I_array).float()
        
        return {'feature': A_tensor, 'dmsp':D_tensor, 'target': B_tensor, 'instance':I_tensor,'feature_paths': A_path}


    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'MultiBandTifDataset'

def CreateMultiBandTifDataset(opt):
    dataset = MultiBandTifDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class MultiBandTifDataLoader(BaseDataLoader):
    def name(self):
        return 'TifDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateMultiBandTifDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)