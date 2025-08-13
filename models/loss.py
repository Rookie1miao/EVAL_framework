import torch

loss_dict = {}

def register_loss(name):
   def decorator(cls):
       #print(f"registering loss {name}")
       loss_dict[name] = cls
       return cls
   return decorator

@register_loss("L1Loss")
class L1Loss(torch.nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        
    def forward(self, outputs, targets):
        m = torch.exp(outputs)
        n = torch.exp(targets)
        l1_loss = torch.mean(torch.abs(m - n))
        
        return l1_loss

@register_loss("ExpMSELoss")
class ExpMSELoss(torch.nn.Module):
    def __init__(self):
        super(ExpMSELoss, self).__init__()
        
    def forward(self, outputs, targets):
        m = torch.exp(outputs) 
        n = torch.exp(targets)

        loss = torch.mean((m - n) ** 2)
        return loss