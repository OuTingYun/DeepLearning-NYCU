import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion, finn_eval_seq,normalize_data,mse_metric,pred,plot_pred
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=1, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=75, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=True, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=1, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=4, help='The number of cycle for kl annealing during training (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=128, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=64, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--p_dim', type=int, default=7, help='position and action dim')

    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')  

    args = parser.parse_args()
    return args

def train(x, cond, modules, optimizer, kl_anneal, args):
    modules['frame_predictor'].zero_grad()
    modules['posterior'].zero_grad()
    modules['encoder'].zero_grad()
    modules['decoder'].zero_grad()

    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    mse = 0
    kld = 0

    x = x.to(torch.device('cuda'))
    cond = cond.to(torch.device('cuda'))
    x = normalize_data(x)
    cond = normalize_data(cond)

    """
    x = [30,12,3,64,64] [time,batch,channel,w,h]
    """

    use_teacher_forcing = True if random.random() < args.tfr else False
    x_pred_list=[x[0]]

    for i in range(1, args.n_past + args.n_future):
        
        if use_teacher_forcing:
            x_t_last = x[i-1]
        else:
            x_t_last = x_pred_list[i-1]


        h_t,_ = modules['encoder'](x[i])

        if i < args.n_past:
            h_t_last,skip = modules['encoder'](x_t_last)
        else:
            h_t_last,_ = modules['encoder'](x_t_last)
        
        # Inference Network
        z_t,mu,logvar = modules['posterior'](h_t)

        # position
        g_t = modules['frame_predictor'](torch.cat([cond[i],h_t_last, z_t], 1))
        x_pred = modules['decoder']([g_t, skip])

        x_pred_list.append(x_pred)

        mse += modules['mseloss'](x_pred,x[i])
        kld += kl_criterion(mu, logvar,args)

    _,_,psnr = finn_eval_seq(x[args.n_past:args.n_future],x_pred_list[args.n_past:args.n_future])
    
    beta = kl_anneal.get_beta()
    loss = mse + kld * beta
    loss.backward()

    optimizer.step()

    return loss.detach().cpu().numpy() / (args.n_past + args.n_future), mse.detach().cpu().numpy() / (args.n_past + args.n_future), kld.detach().cpu().numpy() / (args.n_future + args.n_past),psnr

class kl_annealing():
    def __init__(self, args):
        super().__init__()
        self.n_epoch = args.niter*args.epoch_size
        self.n_cycle = args.kl_anneal_cycle
        self.ratio = args.kl_anneal_ratio
        self.cyclical = args.kl_anneal_cyclical

        
        L = np.ones(self.n_epoch)
        start,stop = 0,1
        if self.cyclical:
            period = self.n_epoch/self.n_cycle
            step = (stop-start)/(period*self.ratio) # linear schedule

            for c in range(self.n_cycle):

                v , i = start , 0
                while v <= stop and (int(i+c*period) < self.n_epoch):
                    L[int(i+c*period)] = v
                    v += step
                    i += 1
        self.beta_cyc = L
            

        self.now = -1

    def update(self):
        self.now+=1
        return

    def get_beta(self):
        self.update()
        return self.beta_cyc[self.now]


def main():
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'
    
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

  
    if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        model_dir = args.model_dir
        optimizer = args.optimizer
        args = saved_model['args']
        
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
        niter = args.niter - start_epoch
        print(niter)
    else:
      
        name = f'niter={args.niter}_epoch={args.epoch_size}_bs={args.batch_size}_kl={args.kl_anneal_cyclical}_klr={args.kl_anneal_ratio}_klc={args.kl_anneal_cycle}_tfrs={args.tfr_start_decay_epoch}'
        args.log_dir = '%s/%s' % (args.log_dir, name)
        niter = args.niter
        start_epoch = 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
        os.remove('./{}/train_record.txt'.format(args.log_dir))
    

    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    # ------------ build the models  --------------

    if args.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
    else:
        frame_predictor = lstm(args.g_dim+args.z_dim+args.p_dim, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
        posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
        frame_predictor.apply(init_weights)
        posterior.apply(init_weights)
            
    if args.model_dir != '':
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
    else:
        encoder = vgg_encoder(args.g_dim)
        decoder = vgg_decoder(args.g_dim)
        encoder.apply(init_weights)
        decoder.apply(init_weights)
    if args.model_dir != '':
        mse_criterion = saved_model['mseloss']
    else:
        mse_criterion = nn.MSELoss()

    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)
    mse_criterion.to(device)

    # --------- load a dataset ------------------------------------
    train_data = bair_robot_pushing_dataset(args, 'train')
    validate_data = bair_robot_pushing_dataset(args, 'validate')
    train_loader = DataLoader(train_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    train_iterator = iter(train_loader)

    validate_loader = DataLoader(validate_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    validate_iterator = iter(validate_loader)

    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
    kl_anneal = kl_annealing(args)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
        'mseloss':mse_criterion
    }
    # --------- training loop ------------------------------------

    if args.model_dir != '':
        best_val_psnr = saved_model['best']
    else:
        best_val_psnr = 0

    print("past best psnr:",best_val_psnr)
        
    if args.tfr_decay_step == 0:
        args.tfr_decay_step = 1/(niter-args.tfr_start_decay_epoch)

    for epoch in range(start_epoch, start_epoch + niter):

        frame_predictor.train()
        posterior.train()
        encoder.train()
        decoder.train()

        epoch_loss = 0
        epoch_mse = 0
        epoch_kld = 0
        epoch_psnr = 0

        for r in  tqdm(range(args.epoch_size),ncols=100):
            try:
                seq, cond = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                seq, cond = next(train_iterator)
            
            loss, mse, kld, psnr = train(seq, cond, modules, optimizer, kl_anneal, args)
            epoch_loss += loss
            epoch_mse += mse
            epoch_kld += kld
            epoch_psnr+=psnr
            

        if epoch >= args.tfr_start_decay_epoch:
            ### Update teacher forcing ratio ###
            args.tfr = 1 - args.tfr_decay_step*(epoch - args.tfr_start_decay_epoch)
            print("tfr:",args.tfr)

        print('[%02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f | psnr: %.5f (%d) ' % (epoch,epoch_loss, epoch_mse,epoch_kld,np.mean(epoch_psnr)/args.epoch_size,(epoch+1)*args.epoch_size*args.batch_size))

        with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
            train_record.write(('[%02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f | psnr: %.5f (%d)\n' % (epoch,epoch_loss, epoch_mse,epoch_kld,np.mean(epoch_psnr)/args.epoch_size,(epoch+1)*args.epoch_size*args.batch_size)))
        
        frame_predictor.eval()
        encoder.eval()
        decoder.eval()
        posterior.eval()

        if epoch % 5 == 0:
            psnr_list = []

            for _ in range(len(validate_data) // args.batch_size):
                try:
                    validate_seq, validate_cond = next(validate_iterator)
                except StopIteration:
                    validate_iterator = iter(validate_loader)
                    validate_seq, validate_cond = next(validate_iterator)
                
                validate_seq = normalize_data(validate_seq.to(torch.device('cuda')))
                validate_cond = normalize_data(validate_cond.to(torch.device('cuda')))

                pred_seq = pred(validate_seq, validate_cond, modules, args)
                _, _, psnr = finn_eval_seq(validate_seq[args.n_past:args.n_past+args.n_future], pred_seq[args.n_past:args.n_past+args.n_future])
                psnr_list.append(psnr)

            try:
                validate_seq, validate_cond = next(validate_iterator)
            except StopIteration:
                validate_iterator = iter(validate_loader)
                validate_seq, validate_cond = next(validate_iterator)
            
            validate_seq = normalize_data(validate_seq.to(torch.device('cuda')))
            validate_cond = normalize_data(validate_cond.to(torch.device('cuda')))
            pred_seq = pred(validate_seq, validate_cond, modules, args)
                    
            plot_pred(validate_seq,pred_seq,epoch,args)
                
            ave_psnr = np.mean(np.concatenate(psnr))
            print('[psnr]: %.5f' % ave_psnr)

            with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
                train_record.write(('====================== validate psnr = {:.5f} ========================\n'.format(ave_psnr)))

            if ave_psnr > best_val_psnr:
                
                best_val_psnr = ave_psnr
                # save the model
                torch.save({
                    'encoder': encoder,
                    'decoder': decoder,
                    'frame_predictor': frame_predictor,
                    'posterior': posterior,
                    'mseloss':mse_criterion,
                    'args': args,
                    'last_epoch': epoch,
                    "best":best_val_psnr
                    },
                    '%s/model.pth' % (args.log_dir))

                print("update model!")

        if epoch%20==0:
            
            torch.save({
                'encoder': encoder,
                'decoder': decoder,
                'frame_predictor': frame_predictor,
                'posterior': posterior,
                'mseloss':mse_criterion,
                'args': args,
                'last_epoch': epoch,
                "best":best_val_psnr
                },
            '%s/model%d.pth' % (args.log_dir,epoch))
            print("normal save.")
       
if __name__ == '__main__':
    main()
        
