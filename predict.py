import yaml
import torch
import argparse
import os
import numpy as np
import rasterio
from models import get_model
from dataloader.data_loader import CreateDataLoader
from dataloader.base_dataset import BaseDataset


class TestOptions:
    def __init__(self):
        self.dataroot = './datasets'      # 数据根目录
        self.phase = 'test'               # 阶段：train/test
        self.isTrain = False              # 是否为训练模式
        self.input_nc = 7                 # 输入特征波段数
        self.output_nc = 1                # 目标数据波段数
        self.batchSize = 1                # 批次大小
        self.serial_batches = True        # 是否按顺序加载
        self.nThreads = 1                 # 数据加载线程数
        self.max_dataset_size = float("inf")  # 最大数据集大小
        self.isInstance = False           # 是否使用实例数据


def predict(model, dataloader, device, output_dir):
    model.to(device)
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = batch['feature'].to(device)
            dmsp = batch['dmsp'].to(device)
            #instances = batch['instance'].to(device)                                                            
            outputs = model(dmsp, inputs)
            #outputs = model(inputs, instance)
            paths = batch['feature_paths']
            outputs_np = outputs.cpu().numpy()

            for j, path in enumerate(paths):
                filename = os.path.basename(path)
                output_path = os.path.join(output_dir, f"{filename}")

                with rasterio.open(path) as src:
                    meta = src.meta.copy()
                
                meta.update({
                    'count': 1,  
                    'dtype': 'float32'
                })
                
                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(outputs_np[j][0], 1) 
                
                print(f"Saved prediction {i*dataloader.batch_size+j+1} to {output_path}")


def main(args):
    config_path = args.config
    device = args.device

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    opt = TestOptions()

    opt.dataroot = args.dataroot if args.dataroot else opt.dataroot

    data_loader = CreateDataLoader(opt)
    dataloader = data_loader.load_data()
    print("DataLoader initialized successfully!")
    # 初始化模型
    model = get_model(**config['model'])
    model_dir = config["training"].get("save_dir", "model_checkpoints")

    ckpt_name = args.checkpoint
    if not ckpt_name:
        ckpt_name = config["training"].get("save_ckpt_name", None)
        if not ckpt_name:
            raise ValueError("请指定要加载的检查点名称")
        ckpt_name = f"{ckpt_name}.pth"
    
    checkpoint_path = os.path.join(ckpt_name)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        filtered_state_dict = {}
        for key, value in checkpoint.items():
            if 'channel_adapter' not in key:
                filtered_state_dict[key] = value

        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"模型加载成功: {checkpoint_path}")

        filtered_keys = [key for key in checkpoint.keys() if 'channel_adapter' in key]
        if filtered_keys:
            print(f"已过滤掉以下动态创建的层: {filtered_keys}")
    else:
        raise FileNotFoundError(f"找不到检查点: {checkpoint_path}")

    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join("predictions", f"{os.path.splitext(ckpt_name)[0]}")

    predict(model, dataloader, device, output_dir)
    print(f"所有预测已保存到: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用训练好的模型进行预测")
    parser.add_argument("--config", type=str, default="configs/.yaml", help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, default='saved_models/.pth', help="模型检查点文件名")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="使用的设备")
    parser.add_argument("--dataroot", type=str, default='', help="测试数据的根目录")
    parser.add_argument("--output_dir", type=str, default='', help="输出预测结果的目录")
    
    args = parser.parse_args()
    
    main(args)