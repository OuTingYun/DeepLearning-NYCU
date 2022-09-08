from evaluator import evaluation_model
from model import Generator,Discriminator,W_Generator,W_Discriminator,DC_Generator,DC_Discriminator

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from torch.optim import Adam
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch

from pathlib import Path
import argparse
import random
import os
import utils
import time

def parse_option():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bs',type=int,default=64,help="batch size")
    parser.add_argument('--im-size',type=int,default=64,help="image size")
    parser.add_argument('--z-size',type=int,default=128,help="latent size")
    parser.add_argument('--g-conv-dim',type=int,default=300,help="generator convolution size")
    parser.add_argument('--d-conv-dim',type=int,default=100,help="discriminator convolution size")
    parser.add_argument('--device',type=str,default="cuda:0",help='cuda or cpu device')
    parser.add_argument('--g-lr',type=float,default=0.0001,help='initial generator learing rate')
    parser.add_argument('--d-lr',type=float,default=0.0004,help='initial discriminator learning rate')
    parser.add_argument('--beta1',type=float,default=0,help='Adam beta 1')
    parser.add_argument('--beta2',type=float,default=0.9,help='Adam beta 2')
    parser.add_argument('--epochs',type=int,default=600,help="total epochs")
    parser.add_argument('--epoch_start',type=int,default=0,help="total epochs")
    parser.add_argument('--eval-iter',type=int,default=50)
    parser.add_argument('--num-cond',type=int,default=24,help='number of conditions')
    parser.add_argument('--adv-loss',type=str,default='wgan-gp',help='adversarial loss method: [bce,hinge,wgan-gp]')
    parser.add_argument('--c-size',type=int,default=100)
    parser.add_argument('--lambda-gp',type=float,default=10)
    parser.add_argument('--net',type=str,default='wgan',help='model')
    parser.add_argument('--model_dir',type=str,default='',help='model director')
    parser.add_argument('--iters',type=int,default=0,help='iteration')
    parser.add_argument('--best_acc',type=int,default=0,help='iteration')
    parser.add_argument('--best_new_acc',type=int,default=0,help='iteration')
    parser.add_argument('--test',type=bool,default=False,help='iteration')
    args=parser.parse_args()
    return args

def train(G,D,g_optimizer,d_optimizer,train_loader,test_loader,new_test_loader,args):
    
    eval_model = evaluation_model()
    writer=SummaryWriter(log_dir=f'runs/{task_name}')
   
    # args.epoch_start = 300
    epoch_start = args.epoch_start
    
    print(args)
    for epoch in range(epoch_start,args.epochs+1):
        
        args.epoch_start = epoch
        train_g_loss=0.0
        train_d_loss=0.0

        # for idx,(img,conds) in enumerate(tqdm(train_loader,ncols=10)):
        for idx,(img,conds) in enumerate(train_loader):  
            G.train()
            D.train()

            img=img.to(args.device)
            conds=conds.to(args.device)
            
            # ========== Train Discriminator =========== #
            d_out_real,dr1,dr2 = D(img, conds)
            d_loss_real = -torch.mean(d_out_real)
            
            # apply Gumbel softmax
            z = utils.tensor2var(torch.randn(args.bs, args.z_size))
            fake_images,gf1,gf2 = G(z, conds)
            d_out_fake,df1,df2 = D(fake_images, conds)
            d_loss_fake = torch.mean(d_out_fake)
          
            # backward + optimize
            d_loss = d_loss_real + d_loss_fake

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            d_loss_item = d_loss.item()

            if args.adv_loss == 'wgan-gp':
                # Compute gradient penalty
                alpha = torch.rand(args.bs, 1, 1, 1).to(args.device).expand_as(img)
                interpolated = Variable(alpha * img.data + (1 - alpha) * fake_images.data, requires_grad=True)
                out,_,_ = D(interpolated,conds)
                grad = torch.autograd.grad( outputs=out,
                                            inputs=interpolated,
                                            grad_outputs=torch.ones(out.size()).to(args.device),
                                            retain_graph=True,
                                            create_graph=True,
                                            only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                d_loss_gp =  args.lambda_gp * d_loss_gp
                d_optimizer.zero_grad()
                d_loss_gp.backward()
                d_optimizer.step()
                d_loss_item += d_loss_gp.item()
            
            # ========== Train generator and gumbel ========== #
            # create random noise
            for _ in range(2):
                z = utils.tensor2var(torch.randn(args.bs, args.z_size))
                fake_images,_,_ = G(z, conds)
            
                # compute loss with fake images
                g_out_fake,_,_= D(fake_images, conds)
                g_loss = -torch.mean(g_out_fake)
                
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            args.iters+=1
            utils.plot(
                i = args.iters,
                writer=writer,
                label = 'iters',
                scalar_dict = { 'g_loss':g_loss.item(),
                                'd_loss':d_loss_item})


            
            
            # evaluate per iteration
            if idx%args.eval_iter==0:
                eval_acc = eval(G,test_loader,new_test_loader,eval_model,args,"eval",writer,epoch,d_loss,g_loss)
                print('iters [%02d]  eval_acc: %.5f eval_new_acc: %.5f'% (args.iters,eval_acc[0],eval_acc[1]))
                utils.plot(
                    i = args.iters,
                    writer=writer,
                    label='eval',
                    scalar_dict = { 'g_loss':g_loss.item(),
                                    'd_loss':d_loss_item,
                                    'eval_acc':eval_acc[0],
                                    'eval_new_acc':eval_acc[1] })
                
            train_g_loss += g_loss.item()
            train_d_loss += d_loss_item  

        train_d_loss/=len(train_loader)
        train_g_loss/=len(train_loader)

        # evauate per epoch
        eval_acc = eval(G,test_loader,new_test_loader,eval_model,args,"epoch",writer,epoch,d_loss,g_loss)
        print(f"[{epoch}/{args.epochs}][AvgG: {train_g_loss:.4f}][AvgD: {train_d_loss:.4f}][Acc: {eval_acc[0]:.4f}][New Acc: {eval_acc[1]:.4f}]")
        with open('logs/{}/train_record.txt'.format(args.model_dir), 'a') as train_record:
            train_record.write(f"[{epoch}/{args.epochs}][AvgG: {train_g_loss:.4f}][AvgD: {train_d_loss:.4f}][Acc: {eval_acc[0]:.4f}]][New Acc: {eval_acc[1]:.4f}]")

        utils.plot( 
            i = epoch,
            writer=writer,
            label='epochs',
            scalar_dict = { 'train_loss_g':train_g_loss,
                            'train_loss_d':train_d_loss,
                            'eval_acc':eval_acc[0],
                            'eval_new_acc':eval_acc[1]})
        
        if epoch%20==0:
            utils.save_model(G,D,args,id=epoch)
            print("save normal model!")


def eval(G,test_loader,new_test_loader,eval_model,args,mode,writer,epoch,d_loss,g_loss):
    G.eval()
    gen_imgs = []
    eval_acc_lise = []
    name = ["test","new_test"]
    for data in [test_loader,new_test_loader]:
        gen_image=None
        eval_acc = 0
        with torch.no_grad():
            for idx,conds in enumerate(data):
                conds=conds.to(args.device)
                z = utils.tensor2var(torch.randn(32, args.z_size))

                fake_images,_,_=G(z,conds)
                gen_image=fake_images
                acc=eval_model.eval(fake_images,conds)
                eval_acc+=acc*conds.shape[0]
        eval_acc/=len(data.dataset)
        
        gen_imgs.append(gen_image)
        eval_acc_lise.append(eval_acc)

    
    if eval_acc_lise[0]>args.best_acc and eval_acc_lise[1]>args.best_new_acc:
        args.best_acc=eval_acc_lise[0]
        args.best_new_acc=eval_acc_lise[1]
        utils.save_model(G,D,args,id="")
        print("save best test and new_test model")

    elif eval_acc_lise[0]>args.best_acc:
        args.best_acc=eval_acc_lise[0]
        utils.save_model(G,D,args,id="%.2f"%eval_acc_lise[0])
        print("save best test model")
    
    elif eval_acc_lise[1]>args.best_new_acc:
        args.best_new_acc=eval_acc_lise[1]
        utils.save_model(G,D,args,id="%.2f"%eval_acc_lise[1])
        print("save best new_test model")


    # 寫入txt + 存圖
    if mode=='eval':
        for idx,gen_images in enumerate(gen_imgs):
            gen_images = 0.5*gen_images+0.5
            save_image(gen_images,f"logs/{args.model_dir}/gen/{name[idx]}_ep{epoch}_iter{args.iters}_acc{eval_acc:.4f}.jpg",nrow=8)
        with open('logs/{}/train_record.txt'.format(args.model_dir), 'a') as train_record:
            train_record.write(('iters  [%02d]  d_loss: %.5f | g_loss: %.5f | eval_acc: %.5f | eval_new_acc %.5f \n'% (args.iters,d_loss.item(),g_loss.item(),eval_acc_lise[0],eval_acc_lise[1])))

    elif mode=='epoch':
        for idx,gen_images in enumerate(gen_imgs):
            gen_images = 0.5*gen_images+0.5
            save_image(gen_images,f"logs/{args.model_dir}/gen/{name[idx]}_ep{epoch}_last_acc{eval_acc:.4f}.jpg",nrow=8)
        with open('logs/{}/train_record.txt'.format(args.model_dir), 'a') as train_record:
            train_record.write(('eopoch [%02d]  d_loss: %.5f | g_loss: %.5f | eval_acc: %.5f | eval_new_acc %.5f \n'% (epoch,d_loss.item(),g_loss.item(),eval_acc_lise[0],eval_acc_lise[1])))

    return eval_acc_lise

# def new_test(G,test_loader,args,name):
#     eval_model = evaluation_model()
#     G.eval()
#     gen_images=None
#     eval_acc = 0
#     with torch.no_grad():
#         for idx,conds in enumerate(test_loader):
#             conds=conds.to(args.device)
#             z = utils.tensor2var(torch.randn(32, args.z_size))

#             fake_images,_,_=G(z,conds)
#             gen_images=fake_images
#             acc=eval_model.eval(fake_images,conds)
#             eval_acc+=acc*conds.shape[0]
#     eval_acc/=len(test_loader.dataset)
#     print(eval_acc)
#     gen_images=0.5*gen_images+0.5 # normalization
#     save_image(gen_images,f"logs/(best){args.model_dir}/{name}.jpg",nrow=8)


if __name__=="__main__":
    args=parse_option()
    
     
    # model name
    t_str = time.strftime("%m-%d(%H:%M)", time.localtime())
    task_name=f"{t_str}_{args.net}_{args.adv_loss}"

    
    # dataset
    data_transform=transforms.Compose([
        transforms.Resize((args.im_size,args.im_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    train_loader,test_loader,new_test_loader = utils.get_data(args,T=data_transform)


    # model
    if args.model_dir=='':
        if args.net =='sagan':
            G=Generator(args.bs,args.im_size,args.z_size,args.g_conv_dim,args.num_cond,args.c_size)
            D=Discriminator(args.bs,args.im_size,args.d_conv_dim,args.num_cond)
        elif args.net == 'wgan':
            G=W_Generator(args.bs,args.im_size,args.z_size,args.g_conv_dim,args.num_cond,args.c_size)
            D=W_Discriminator(args.bs,args.im_size,args.d_conv_dim,args.num_cond)
        elif args.net == 'dcgan':
            G=DC_Generator(args.bs,args.im_size,args.z_size,args.g_conv_dim,args.num_cond,args.c_size)
            D=DC_Discriminator(args.bs,args.im_size,args.d_conv_dim,args.num_cond)

        args.device = torch.device(args.device)      
        args.model_dir = task_name 
    
    else:
        
        saved_model = torch.load('logs/%s/model.pth' % args.model_dir)
        args = saved_model["args"]
        G = W_Generator(args.bs,args.im_size,args.z_size,args.g_conv_dim,args.num_cond,args.c_size)
        D = W_Discriminator(args.bs,args.im_size,args.d_conv_dim,args.num_cond)
        G = saved_model["Generator"]
        D = saved_model["Discriminator"]
        
        task_name = args.model_dir
    
    # print(args)
    G=G.to(args.device)
    D=D.to(args.device)
    
    if not args.test:
        # optimizer
        g_optimizer=Adam(filter(lambda p:p.requires_grad,G.parameters()),lr=args.g_lr,betas=[args.beta1,args.beta2])
        d_optimizer=Adam(filter(lambda p:p.requires_grad,D.parameters()),lr=args.d_lr,betas=[args.beta1,args.beta2])
        # record
        os.makedirs("logs", exist_ok=True)
        os.makedirs('logs/%s/gen/' % args.model_dir, exist_ok=True)

        with open('logs/{}/train_record.txt'.format(args.model_dir), 'a') as train_record:
            train_record.write('args: {}\n'.format(args))

        
        train(G=G,
            D=D,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
            new_test_loader = new_test_loader,
            args=args,
            )
    # else:
    # print("testing")
    # new_test(G,new_test_loader,args,"new_test")
    # new_test(G,test_loader,args,"test")