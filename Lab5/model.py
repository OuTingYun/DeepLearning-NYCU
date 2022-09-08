import torch.nn as nn
from  spectral import SpectralNorm
import torch
import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class Generator(nn.Module):
    """Generator."""

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64,num_cond=24,c_size=100):
        super(Generator, self).__init__()
        self.imsize = image_size
        self.c_size=c_size
        self.embed_c=nn.Sequential(
            nn.Linear(num_cond,c_size),
            nn.ReLU(inplace=True)
        )

        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3  # 6-3=3
        mult = 2 ** repeat_num # 8

        # h_new = (h-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim+c_size, conv_dim * mult, kernel_size=4)))   # 1 -> 4
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), kernel_size=4, stride=2, padding=1)))    # 4 -> 8(6-2+3+0+1)
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), kernel_size=4, stride=2, padding=1)))   # 8 -> 16
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), kernel_size=4, stride=2, padding=1)))   # 16 -> 32
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))   # 32 -> 64
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn( conv_dim*2, 'relu') #8/2/2
        self.attn2 = Self_Attn( conv_dim,  'relu')  #8/2/2/2

    def forward(self, z,c):
        z = z.view(z.size(0), z.size(1), 1, 1)  #[batch,z_size,1,1] = [bs,128,1,1]
        c_embd=self.embed_c(c).reshape(-1,self.c_size,1,1)  # [bs,1]
        z=torch.cat((z,c_embd),dim=1) # [12,c+z,1,1]
        out=self.l1(z)  # 1 -> 4        [12,2400]
        out=self.l2(out)    # 4 -> 8    [12,1200]
        out=self.l3(out)    # 8 -> 16   [12,600]    
        out,p1 = self.attn1(out)
        out=self.l4(out)    # 16 -> 32  [12,300]    
        out,p2 = self.attn2(out)
        out=self.last(out)  # 32 -> 64  [12,3]

        return out, p1, p2


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64,num_cond=24):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        self.embed_c=nn.Sequential(
            nn.Linear(num_cond,self.imsize*self.imsize),
            nn.ReLU(inplace=True)
        )

        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(4, conv_dim, kernel_size=4, stride=2, padding=1)))
        layer1.append(nn.LeakyReLU(0.1))
        # layer1.append(nn.ReLU())

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(conv_dim*4, 'relu')
        self.attn2 = Self_Attn(conv_dim*8, 'relu')

    def forward(self, x,c):
        c_embd=self.embed_c(c).reshape(-1,1,self.imsize,self.imsize)
        x=torch.cat((x,c_embd),dim=1)   # [12, 4, 64, 64]
        out = self.l1(x)    # [12, 100, 32, 32]
        out = self.l2(out)  # [12, 200, 16, 16]
        out = self.l3(out)  # [12, 400, 8, 8]
        out,p1 = self.attn1(out)    
        out=self.l4(out)    # [12, 800, 4, 4]
        out,p2 = self.attn2(out)
        out=self.last(out)  # [12, 1, 1, 1]
        return out.squeeze(), p1, p2
    
class W_Generator(nn.Module):

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64,num_cond=24,c_size=100):
        super(W_Generator, self).__init__()
        self.imsize = image_size
        self.c_size=c_size
        self.embed_c=nn.Sequential(
            nn.Linear(num_cond,c_size),
            nn.ReLU(inplace=True)
        )


        repeat_num = int(np.log2(self.imsize)) - 3  # 6-3=3
        mult = 2 ** repeat_num # 8

        # h_new = (h-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1
        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim+c_size, conv_dim * mult, kernel_size=4),   # 1 -> 4
            nn.BatchNorm2d(conv_dim * mult),
            nn.ReLU(),
        )

        curr_dim = conv_dim * mult

        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), kernel_size=4, stride=2, padding=1),    # 4 -> 8(6-2+3+0+1)
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(),
         )

        curr_dim = int(curr_dim / 2)

        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), kernel_size=4, stride=2, padding=1),   # 8 -> 16
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(),
        )

        curr_dim = int(curr_dim / 2)

        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(),

        )

        curr_dim = int(curr_dim / 2)
        self.last = nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1)# 32 -> 64
    
    def forward(self, z,c):
        p1,p2 = 2,2
        z = z.view(z.size(0), z.size(1), 1, 1)  #[batch,z_size,1,1] = [bs,128,1,1]
        c_embd=self.embed_c(c).reshape(-1,self.c_size,1,1)  # [bs,1]
        z=torch.cat((z,c_embd),dim=1) # [12,c+z,1,1]
        out=self.l1(z)  # 1 -> 4        [12,2400]
        out=self.l2(out)    # 4 -> 8    [12,1200]
        out=self.l3(out)    # 8 -> 16   [12,600]    
        out=self.l4(out)    # 16 -> 32  [12,300]    
        out=self.last(out)  # 32 -> 64  [12,3]

        return out, p1, p2


class W_Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64,num_cond=24):
        super(W_Discriminator, self).__init__()
        self.imsize = image_size
        self.embed_c=nn.Sequential(
            nn.Linear(num_cond,self.imsize*self.imsize),
            nn.ReLU(inplace=True)
        )


        self.l1 = nn.Sequential(
            nn.Conv2d(4, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        self.l2 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.LeakyReLU(0.1)

        )

        curr_dim = curr_dim * 2

        self.l3 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.LeakyReLU(0.1)
        )
        curr_dim = curr_dim * 2

        self.l4 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.LeakyReLU(0.1)
        )
        curr_dim = curr_dim*2

        self.last = nn.Conv2d(curr_dim, 1, 4)

    def forward(self, x,c):

        p1, p2 = 1,1
        c_embd=self.embed_c(c).reshape(-1,1,self.imsize,self.imsize)
        x=torch.cat((x,c_embd),dim=1)   # [12, 4, 64, 64]
        out = self.l1(x)    # [12, 100, 32, 32]
        out = self.l2(out)  # [12, 200, 16, 16]
        out = self.l3(out)  # [12, 400, 8, 8]
        out=self.l4(out)    # [12, 800, 4, 4]
        out=self.last(out)  # [12, 1, 1, 1]
        return out.squeeze(), p1, p2
    
class DC_Generator(nn.Module):

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64,num_cond=24,c_size=100):
        super(DC_Generator, self).__init__()
        self.imsize = image_size
        self.c_size=c_size
        self.embed_c=nn.Sequential(
            nn.Linear(num_cond,c_size),
            nn.ReLU(inplace=True)
        )


        repeat_num = int(np.log2(self.imsize)) - 3  # 6-3=3
        mult = 2 ** repeat_num # 8

        # h_new = (h-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1
        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim+c_size, conv_dim * mult, kernel_size=4),   # 1 -> 4
            nn.BatchNorm2d(conv_dim * mult),
            nn.ReLU(),
        )

        curr_dim = conv_dim * mult

        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), kernel_size=4, stride=2, padding=1),    # 4 -> 8(6-2+3+0+1)
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(),
         )

        curr_dim = int(curr_dim / 2)

        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), kernel_size=4, stride=2, padding=1),   # 8 -> 16
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(),
        )

        curr_dim = int(curr_dim / 2)

        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(),

        )

        curr_dim = int(curr_dim / 2)
        self.last = nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1)# 32 -> 64
        self.tanh=nn.Tanh()
    def forward(self, z,c):
        p1,p2 = 2,2
        z = z.view(z.size(0), z.size(1), 1, 1)  #[batch,z_size,1,1] = [bs,128,1,1]
        c_embd=self.embed_c(c).reshape(-1,self.c_size,1,1)  # [bs,1]
        z=torch.cat((z,c_embd),dim=1) # [12,c+z,1,1]
        out=self.l1(z)  # 1 -> 4        [12,2400]
        out=self.l2(out)    # 4 -> 8    [12,1200]
        out=self.l3(out)    # 8 -> 16   [12,600]    
        out=self.l4(out)    # 16 -> 32  [12,300]    
        out=self.last(out)  # 32 -> 64  [12,3]
        out=self.tanh(out)
        return out, p1, p2


class DC_Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64,num_cond=24):
        super(DC_Discriminator, self).__init__()
        self.imsize = image_size
        self.embed_c=nn.Sequential(
            nn.Linear(num_cond,self.imsize*self.imsize),
            nn.ReLU(inplace=True)
        )


        self.l1 = nn.Sequential(
            nn.Conv2d(4, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU())

        curr_dim = conv_dim

        self.l2 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(curr_dim * 2),
            nn.LeakyReLU()

        )

        curr_dim = curr_dim * 2

        self.l3 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(curr_dim * 2),
            nn.LeakyReLU()
        )
        curr_dim = curr_dim * 2

        self.l4 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(curr_dim * 2),
            nn.LeakyReLU()
        )
        curr_dim = curr_dim*2

        self.last = nn.Conv2d(curr_dim, 1, 4)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x,c):

        p1, p2 = 1,1
        c_embd=self.embed_c(c).reshape(-1,1,self.imsize,self.imsize)
        x=torch.cat((x,c_embd),dim=1)   # [12, 4, 64, 64]
        out = self.l1(x)    # [12, 100, 32, 32]
        out = self.l2(out)  # [12, 200, 16, 16]
        out = self.l3(out)  # [12, 400, 8, 8]
        out=self.l4(out)    # [12, 800, 4, 4]
        out=self.last(out)  # [12, 1, 1, 1]
        out =self.sigmoid(out)
        return out.view(-1), p1, p2
    