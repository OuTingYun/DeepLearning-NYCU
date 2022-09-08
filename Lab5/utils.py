from email.generator import Generator
from lib2to3.pgen2.pgen import generate_grammar
from dataloader import ICLEVRLoader
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

def tensor2var(x,device,grad=False):
    x=x.to(device)
    x=Variable(x,requires_grad=grad)
    return x
def get_data(args,T):
    train_set=ICLEVRLoader(trans=T,mode='train')
    train_loader=DataLoader(train_set,
        batch_size=args.bs,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )
    test_set=ICLEVRLoader(mode='test')
    test_loader=DataLoader(test_set,
    batch_size=args.bs,
    shuffle=False,
    num_workers=4,
    pin_memory=True
    )
    return train_loader,test_loader

def save_model(G,D,args,id=""):
    torch.save({
        "Generator":G,
        "Discriminator":D,
        "args":args
        },
    f'logs/{args.model_dir}/model{id}.pth')

def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)

def plot(label,scalar_dict,i,writer):
    for key,val in scalar_dict.items():
        writer.add_scalar(f"{label}/{key}",val,i)
    

