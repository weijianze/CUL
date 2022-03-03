import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import pdb

def gabor_fn(kernel_size, channel_in, channel_out, sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma    # [channel_out]
    sigma_y = sigma.float() / gamma     # element-wize division, [channel_out]

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = kernel_size // 2
    ymax = kernel_size // 2
    xmin = -xmax
    ymin = -ymax
    ksize = xmax - xmin + 1
    y_0 = torch.arange(ymin, ymax+1)
    y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float()
    x_0 = torch.arange(xmin, xmax+1)
    x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize).float()   # [channel_out, channelin, kernel, kernel]
    if theta.is_cuda:
        x = x.cuda()
        y = y.cuda()

    # Rotation
    # don't need to expand, use broadcasting, [64, 1, 1, 1] + [64, 3, 7, 7]
    # pdb.set_trace()
    x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
    y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

    # [channel_out, channel_in, kernel, kernel]
    gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1, 1) ** 2)) \
         * torch.cos(2 * math.pi / Lambda.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))

    return gb


class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0):
        super(GaborConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding

        self.Lambda = nn.Parameter(torch.rand(channel_out), requires_grad=True)
        self.theta = nn.Parameter(torch.randn(channel_out) * 1.0, requires_grad=True)
        self.psi = nn.Parameter(torch.randn(channel_out) * 0.02, requires_grad=True)
        self.sigma = nn.Parameter(torch.randn(channel_out) * 1.0, requires_grad=True)
        self.gamma = nn.Parameter(torch.randn(channel_out) * 0.0, requires_grad=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        theta = self.sigmoid(self.theta) * math.pi * 2.0
        gamma = 1.0 + (self.gamma * 0.5)
        sigma = 0.1 + (self.sigmoid(self.sigma) * 0.4)
        Lambda = 0.001 + (self.sigmoid(self.Lambda) * 0.999)
        psi = self.psi

        kernel = gabor_fn(self.kernel_size, self.channel_in, self.channel_out, sigma, theta, Lambda, psi, gamma)
        kernel = kernel.float()   # [channel_out, channel_in, kernel, kernel]
        # if x.is_cuda():
        # pdb.set_trace()

        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)

        return out


def getGaborFilterBank(nScale, M, h, w):#nscale表示波长，M表示方向，h表示图片的长，w表示图片的宽
    Kmax = math.pi / 2
    f = math.sqrt(2)
    sigma = math.pi
    sqsigma = sigma ** 2
    postmean = math.exp(-sqsigma / 2)
    if h != 1:
        gfilter_real = torch.zeros(M, h, w)
        for i in range(M):
            theta = i / M * math.pi
            k = Kmax / f ** (nScale - 1)
            xymax = -1e309
            xymin = 1e309
            for y in range(h):
                for x in range(w):
                    y1 = y + 1 - ((h + 1) / 2)
                    x1 = x + 1 - ((w + 1) / 2)
                    tmp1 = math.exp(-(k * k * (x1 * x1 + y1 * y1) / (2 * sqsigma)))
                    #tmp2 = math.cos(k * math.cos(theta) * x1 + k * math.sin(theta) * y1) - postmean # For real part
                    tmp2 = math.sin(k * math.cos(theta) * x1 + k * math.sin(theta) * y1) # For imaginary part
                    gfilter_real[i][y][x] = k * k * tmp1 * tmp2 / sqsigma			
                    xymax = max(xymax, gfilter_real[i][y][x])
                    xymin = min(xymin, gfilter_real[i][y][x])
            gfilter_real[i] = (gfilter_real[i] - xymin) / (xymax - xymin)
    else:
        gfilter_real = torch.ones(M, h, w)
    return gfilter_real

class GaborConv2d_math(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, orient=4, scale=3, padding=0):
        super(GaborConv2d_math, self).__init__()
        self.padding = padding
        self.kernel = getGaborFilterBank(scale,channel_out,kernel_size, kernel_size).unsqueeze(1).repeat(1,channel_in,1,1)

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding = self.padding)


if __name__=='__main__':
    x = torch.randn(10,1,128,128)
    net_gabor = GaborConv2d(1,6,7,padding=3)
    net_gabor_math = GaborConv2d_math(1,6,7,padding=3)

    print(net_gabor(x).size())
    print(net_gabor_math(x).size())