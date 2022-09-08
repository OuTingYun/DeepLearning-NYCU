import pandas as pd
from torch.utils.data import DataLoader,Dataset
import numpy as np
import PIL.Image as Image
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
import torch
import torchvision.models

import matplotlib.pyplot as plt


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(Dataset):
    def __init__(self, root, mode,transform=None):
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        if transform == None:
            self.transform = transforms.ToTensor()
        else: self.transform = transform
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        
        path = self.root + self.img_name[index] + '.jpeg'
        img = Image.open(path , mode='r')

        if self.transform!=None:
            img = self.transform(img)

        label = self.label[index]

        return img, label

class ResNet(nn.Module):
    def __init__(self, pretrained=True,layers=18):
        super(ResNet, self).__init__()

        f = lambda x : "with" if x else "w/o" 
        self.pretrained = f(pretrained)
        self.pretrained_bool = pretrained
        self.layer_num = layers
        

        pretrained_model = torchvision.models.__dict__[f'resnet{layers}'](pretrained)
        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']


        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classify = nn.Sequential(
            nn.Linear(pretrained_model._modules['fc'].in_features, 5),
            # nn.Dropout(),
            # nn.Linear(256, 5),
            # nn.Softmax(dim=1)
        )
        print("classify",self.classify)

        del pretrained_model

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)   #[4*512*1*1]=[樣本數*neurons]
        x = self.classify(x)
        return x

def show_result(Acc_list,layer):
    def show_acc(acc,name):
        plt.plot(acc,label=name) 
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy%')
    plt.title(f'Result comparison:({layer})')
    
    print(Acc_list)
    for t in ["with"]:
        for pre in ["train","test"]:
            show_acc(Acc_list[t][pre],f"{pre}({t}) pretraining")      
    plt.legend()

    plt.savefig(f'resnet{layer}.png')

    for t in ["with"]:
        for pre in ["train","test"]:
            print(f"{pre}({t}):{max(Acc_list[t][pre])}%")

def train(model,device,epochs,Loss,optimizer,train_data,test_data,train_size,test_size):
    
    print(f"loss:{Loss},op:{optimizer}")

    model = model.to(device)
    get80,get120 = 0,0

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classify.parameters():
        param.requires_grad = True

    for epoch in range(epochs):
        
        if epoch==2:
            for param in model.parameters():
                param.requires_grad = True

        print(f"Epoch:{epoch:2d}   ...")

        train_acc_list, test_acc_list, train_acc, test_acc,train_loss,test_loss  = [], [], 0, 0,0,0
        model.train()

        for idx,(data,target) in enumerate(tqdm(train_data)):
            
            data=data.to(device,dtype=torch.float)
            target=target.to(device,dtype=torch.long)

            predict = model(data)
            loss = Loss(predict,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc  += (torch.max(predict,dim=1).indices==target).sum().item()

            # break
        model.eval()
        with torch.no_grad():
            for idx,(data,target) in enumerate(tqdm(test_data)):
                data = data.to(device,dtype=torch.float)
                target = target.to(device,dtype=torch.float)
                predict = model(data)
                test_acc  += (torch.max(predict,dim=1).indices==target).sum().item()
                # break
            
        
        
        train_acc_list.append(round(100* train_acc/train_size,2))
        test_acc_list.append(round(100* test_acc/test_size,2))
        

        print(f'Epoch:{epoch:2d} Train Acc:{train_acc_list[-1]:3.2f} Test Acc:{test_acc_list[-1]:3.2f} Train Loss:{train_loss:3.2f}')
        

        if test_acc_list[-1]>=82 and test_acc_list[-1]>get120:
            print(f"save model at model3/resnet{model.layer_num}_{model.pretrained_bool}_82.pth")
            torch.save(model.state_dict(), f"model3/resnet{model.layer_num}_{model.pretrained_bool}_82.pth")
            get120 = test_acc_list[-1]
            return {f"{model.pretrained}":{"train":train_acc_list,"test":test_acc_list}}

        elif test_acc_list[-1]>=80 and test_acc_list[-1]>get80:
            print(f"save model at model3/resnet{model.layer_num}_{model.pretrained_bool}_80.pth")
            torch.save(model.state_dict(), f"model3/resnet{model.layer_num}_{model.pretrained_bool}_80.pth")
            get80 = test_acc_list[-1]

    print(f"save model at model3/resnet{model.layer_num}_{model.pretrained_bool}.pth")
    torch.save(model.state_dict(), f"model3/resnet{model.layer_num}_{model.pretrained_bool}.pth")
    return {f"{model.pretrained}":{"train":train_acc_list,"test":test_acc_list}}


            
def LoadData(param):
    print("batch:",param['batch'],"Epochs:",param["epochs"],"lr",param["learning_rate"])
    p = {}
    data_loader = RetinopathyLoader('data/','train',transform=param['transforms'])
    p['train_size'] = len(data_loader)
    p['train_data'] = DataLoader(data_loader, batch_size=param['batch'],shuffle=True)
    
    data_loader = RetinopathyLoader('data/','test',transform=param['transforms'])
    p['test_size'] = len(data_loader)
    p['test_data'] = DataLoader(data_loader, batch_size=param['batch'],shuffle=True)
    
    return p
    
def SetFunction(param,m_type=True):
    
    hyper = param.copy()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hyper['device'] = device
    print(device)

    # 是否取新的model
    if m_type :
        model = ResNet(param['model']['pretrained'],param['model']['layer'])
    else:
        model = ResNet()
        model.load_state_dict(torch.load("./model/resnet59_True.pth"))
    
    hyper['model'] = model

    # 是否更改optimizer
    if param['optimizer']['name'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),lr=param['learning_rate'],momentum=param['optimizer']['momentum'],weight_decay=param['optimizer']['weight_decay'],nesterov=param['optimizer']['nesterov'])
        
    elif param['optimizer']['name'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),weight_decay=param['optimizer']['weight_decay'])
    
    hyper['optimizer'] = optimizer

    # 是否換loss
    if param['Loss']=='CrossEntropy':
        hyper['Loss'] = nn.CrossEntropyLoss()    

    elif param['Loss']=='MSE':
        hyper['Loss'] = nn.MSELoss()

    del hyper['transforms'],hyper['batch'],hyper['learning_rate']

    return hyper
    
if __name__ == "__main__":
    param = { 
        "epochs":30,
        "batch":4,
        "learning_rate":1e-3,
        "Loss":"CrossEntropy",
        "optimizer":{
            "name":"SGD",
            "momentum":0.9,
            "weight_decay":5e-4,
            "nesterov":False #
        },
        'model':{
            'layer':18,
            'pretrained':True
        },
        'transforms':transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.1, 0.1, 0.1]), 
        ])
    }

    data = LoadData(param)

    for layer in [18,50]:
        resnet = f"ResNet{layer}"
        Acc_list = {resnet:{}}
        for pretrained in [True,False]:
            print(f"{resnet}  ({pretrained})")
           
            
            param['model'] = {'pretrained': pretrained,'layer':layer}
            
            hyper = {**data,**SetFunction(param)}        
            Acc_list[resnet] = {**Acc_list[resnet],** train(**hyper)}
            
            show_result(Acc_list[resnet],layer)
