import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, random_split
import argparse
from models import get_model
from dataloader.data_loader import CreateDataLoader
import time
import os
from models.loss import loss_dict
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm


class Options:
    """模拟数据加载器所需选项"""
    def __init__(self,
                 dataroot='./datasets',
                 phase='train',
                 isTrain=True,
                 input_nc=1,
                 output_nc=1,
                 batchSize=2,
                 serial_batches=False,
                 nThreads=2,
                 max_dataset_size=float("inf"),
                 val_ratio=0.2,
                 seed=22,
                 **kwargs):
        self.dataroot = dataroot
        self.phase = phase
        self.isTrain = isTrain
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.batchSize = batchSize
        self.serial_batches = serial_batches
        self.nThreads = nThreads
        self.max_dataset_size = max_dataset_size
        self.kwargs = kwargs
        self.val_ratio = val_ratio
        self.seed = seed 

def train_model(model, dataloader, optimizer, criterion, device, num_epochs=25, scheduler=None, val_loader=None):
    model.to(device)
    
    train_loss, val_loss = [], []
    best_val_loss = float('inf')
    best_state_dict = None
    best_epoch = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(dataloader, leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")

        for batch in loop:
            inputs = batch['feature'].to(device)
            dmsp = batch['dmsp'].to(device)
            targets = batch['target'].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(dmsp, inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataloader.dataset)
        train_loss.append(epoch_loss)

        if val_loader is not None:
            model.eval()
            running_val = 0.0
            for batch in val_loader:
                inputs = batch['feature'].to(device)
                dmsp = batch['dmsp'].to(device)
                targets = batch['target'].to(device)
                outputs = model(dmsp, inputs)
                loss = criterion(outputs, targets)
                
                running_val += loss.item() * inputs.size(0)

            epoch_val_loss = running_val / len(val_loader.dataset)
            val_loss.append(epoch_val_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

            if scheduler is not None:
                scheduler.step(epoch_val_loss)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_state_dict = model.state_dict()
                best_epoch = epoch + 1
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")
            if scheduler is not None:
                scheduler.step(epoch_loss)

    return train_loss, val_loss, best_state_dict, best_epoch

def main(args):
    config_path = args.config
    device = args.device
    continue_path = args.checkpoint
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_dir = config["training"].get("save_dir", "model_checkpoints")

    opt = Options(**config.get("dataloader", {}))
    data_loader = CreateDataLoader(opt)
    dataloader = data_loader.load_data()
    if opt.val_ratio > 0.0:
        val_loader = data_loader.load_val_data()
        print("Validation DataLoader initialized successfully!")
    else:
        val_loader = None
        print("Validation DataLoader not initialized.")
    print("DataLoader initialized successfully!")

    # init model
    model = get_model(**config['model'])
    print("Model initialized successfully!")
    
    if continue_path:
        if os.path.exists(continue_path):
            model.load_state_dict(torch.load(continue_path, map_location=device))
            print(f"Continuing training from checkpoint: {continue_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {continue_path}")

    loss_function_name = config["training"][f"loss_function"]
    try:
        print(f"Using loss function: {loss_function_name}")
        loss_fn = loss_dict[loss_function_name]()
    except KeyError:
        print(f"Loss function {loss_function_name} not found. Using default.")
        loss_fn = getattr(torch.nn, loss_function_name)()

    optimizer = getattr(torch.optim, config["training"]["optimizer"])(
        model.parameters(), lr=config["training"]["learning_rate"]
    )

    if config["training"].get("scheduler", None) == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=config["training"]["patience"], factor=0.5
        )
    else:
        scheduler = None

    num_epochs = config["training"].get("epochs")

    train_loss, val_loss, best_state_dict, best_epoch = train_model(model, dataloader, optimizer, loss_fn, device, num_epochs=num_epochs, scheduler=scheduler, val_loader=val_loader)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    if len(val_loss) == 0:
        plt.plot(train_loss, label='Train Loss')
    else:
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.yscale('log') 
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(model_dir, f"loss_curves.png"))
    plt.close()
    print(f"Loss curves saved at {os.path.join(model_dir, f'loss_curves.png')}")

    # save model
    current_time = time.strftime("%Y%m%d-%H%M%S")
    if not config["training"].get("save_ckpt_name", None):
        save_path = os.path.join(model_dir, f"{config['model']['model_name']}_{config['model']['encoder_name']}_{current_time}.pth")
    else:
        save_path = os.path.join(model_dir, f"{config['training']['save_ckpt_name']}.pth")

    torch.save(model.state_dict(), save_path)
    print(f"model saved at {save_path}")

    # If best model exists (with validation), save it
    if best_state_dict is not None:
        best_save_path = save_path.replace(".pth", f"_best_epoch_{best_epoch}.pth")
        torch.save(best_state_dict, best_save_path)
        print(f"Best model saved at {best_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/S1_config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint", type=str, default=None, help="Continue training from a specific checkpoint")
    args = parser.parse_args()
    
    main(args)