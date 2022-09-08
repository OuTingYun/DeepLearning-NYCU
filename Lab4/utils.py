import math
from operator import pos
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms as T


def kl_criterion(mu, logvar, args):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= args.batch_size  
  return KLD
    
def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c]) 
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]

    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i].detach().cpu().numpy()
            predict = pred[t][i].detach().cpu().numpy()
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def finn_psnr(x, y, data_range=1.):
    mse = ((x - y)**2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def normalize_data(seq):
    seq = seq.transpose(0, 1) #將seq[0] seq[1]轉至，
    return seq

def pred(validate_seq, validate_cond, modules, args):
    x = validate_seq
    cond = validate_cond
    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()

    with torch.no_grad():  
              
        x_pred_list=[x[0]]
        
        for i in range(1, args.n_past + args.n_future):

            x_t_last = x_pred_list[i-1]

            if i < args.n_past:
                h_t_last,skip = modules['encoder'](x_t_last)
            else:
                h_t_last,_ = modules['encoder'](x_t_last)
            
            z_t = torch.cuda.FloatTensor(args.batch_size,args.z_dim).normal_()

            # position
            g_t = modules['frame_predictor'](torch.cat([cond[i],h_t_last, z_t], 1))
            x_pred = modules['decoder']([g_t, skip])
            x_pred_list.append(x_pred)
        
        return x_pred_list


def plot_img(fname,img_seq):
    transform = T.ToPILImage()
    
    sample = len(img_seq)
    time = len(img_seq[0])
    bg = Image.new('RGB',(time*64, sample*64), '#000000')
    for s in range(sample):
        for t in range(time):
            img = transform(img_seq[s][t])  
                 
            x = (t) % time 
            y = (s) % sample                  # 根據開啟的順序，決定 x 座標
            bg.paste(img,(x*64, y*64))   # 貼上圖片

    bg.save(fname)
def plot_gif(fname,img_seq):
    transform = T.ToPILImage()
    sample = len(img_seq)
    time = len(img_seq[0])
    gif = []
    for t in range(time):
        g_bg = Image.new('RGB',(sample*64,64), '#000000')

        for s in range(sample):
            img = transform(img_seq[s][t])    
            x = (s) % 30   
            g_bg.paste(img,(x*64,0))
        gif.append(g_bg)
    g_bg.save(fname, save_all=True, append_images=gif[0:], optimize=False, duration=100, loop=0)

def plot_pred(validate_seq,gen, epoch, args):
    sample = 0
    plot_list = [[],[]]
    for t in range(len(gen)):
        plot_list[0].append(validate_seq[t][sample])
        plot_list[1].append(gen[t][sample])
    plot_img(f"{args.log_dir}/gen/Eopoch_{epoch}.jpg",plot_list)
    plot_gif(f"{args.log_dir}/gen/Epoch_{epoch}.gif",plot_list)
                
        


    


