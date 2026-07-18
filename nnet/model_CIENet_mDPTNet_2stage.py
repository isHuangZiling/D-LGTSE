#Author: Xue Yang
#Date: 2023-05
#Email: yangx11@emails.bjut.edu.cn
#Copyright (c) Institute of Speech and Audio Information Processing, Beijing University of Technology. All rights reserved.
#License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window


class SingleBlock(nn.Module):
    def __init__(self, rnn_type, input_dim, hidden_dim, nhead, dropout=0, bidirectional=True, eps=1e-8):
        super(SingleBlock, self).__init__()

        self.feature_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_direction = int(bidirectional) + 1

        self.att_freq = nn.MultiheadAttention(self.feature_dim, self.nhead, batch_first=True)
        self.norm_att_freq = nn.LayerNorm(self.feature_dim, eps=eps)
        self.rnn_freq = getattr(nn, rnn_type)(self.feature_dim, self.hidden_dim, 1, dropout=dropout,
                                         batch_first=True, bidirectional=bidirectional)
        self.fc_freq = nn.Linear(self.hidden_dim * self.num_direction, self.feature_dim)
        self.norm_freq = nn.LayerNorm(self.feature_dim, eps=eps)

        self.att_time = nn.MultiheadAttention(self.feature_dim, self.nhead, batch_first=True)
        self.norm_att_time = nn.LayerNorm(self.feature_dim, eps=eps)
        self.rnn_time = getattr(nn, rnn_type)(self.feature_dim, self.hidden_dim, 1, dropout=dropout,
                                              batch_first=True, bidirectional=bidirectional)
        self.fc_time = nn.Linear(self.hidden_dim * self.num_direction, self.feature_dim)
        self.norm_time = nn.LayerNorm(self.feature_dim, eps=eps)

    def forward(self, input):
        # input: (Batch, Channel, Freq, Time)

        batch_size, channel, freq, time = input.size()

        input_freq = input.permute(0, 3, 2, 1).contiguous().view(batch_size * time, freq, -1)
        att_out_freq, _ = self.att_freq(input_freq, input_freq, input_freq)
        att_out_freq = att_out_freq + input_freq
        att_out_freq = self.norm_att_freq(att_out_freq)
        rnn_out_freq, _ = self.rnn_freq(att_out_freq)
        fc_out_freq = self.fc_freq(rnn_out_freq)
        fc_out_freq = fc_out_freq + att_out_freq
        fc_out_freq = self.norm_freq(fc_out_freq)
        fc_out_freq = fc_out_freq.contiguous().view(batch_size, time, freq, -1).permute(0, 3, 2, 1)

        input_time = fc_out_freq.permute(0, 2, 3, 1).contiguous().view(batch_size * freq, time, -1)
        att_out_time, _ = self.att_time(input_time, input_time, input_time)
        att_out_time = att_out_time + input_time
        att_out_time = self.norm_att_time(att_out_time)
        rnn_out_time, _ = self.rnn_time(att_out_time)
        fc_out_time = self.fc_time(rnn_out_time)
        fc_out_time = fc_out_time + att_out_time
        fc_out_time = self.norm_time(fc_out_time)
        fc_out_time = fc_out_time.contiguous().view(batch_size, freq, time, -1).permute(0, 3, 1, 2)

        return fc_out_time


class Separator(nn.Module):
    def __init__(self, input_dim, output_dim, bottleneck_dim, hidden_dim, nhead, num_layer, eps=1e-8):
        super(Separator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_layer = num_layer

        self.LayerNormalization = nn.GroupNorm(1, self.input_dim, eps=eps)

        self.bottleneck = nn.Conv2d(self.input_dim, self.bottleneck_dim, 1)

        self.Stack = nn.ModuleList()
        for i in range(num_layer):
            self.Stack.append(SingleBlock('LSTM', self.bottleneck_dim, hidden_dim, nhead))

        self.output_con2d = nn.Conv2d(self.bottleneck_dim, self.output_dim, 1)


    def forward(self, input):
        # input size : (Batch, Channel, Freq, Time)

        output = self.bottleneck(self.LayerNormalization(input))

        for i in range(len(self.Stack)):
            output = self.Stack[i](output)

        output = self.output_con2d(output)

        return output


class FreqNet(nn.Module):
    def __init__(self, enc_dim=256, bottleneck_dim=64, hidden_dim=128, nhead=4, num_layer=6,
                 sr=8000, win_size=256, fft_len=256, win_type='hann'):
        super(FreqNet, self).__init__()

        self.enc_dim = enc_dim
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.nhead = nhead
        self.sr = sr
        self.win_size = win_size
        self.hop_size = win_size // 2

        self.fft_len = fft_len
        self.win_type = win_type

        self.softmax = nn.Softmax(dim=-2)

        self.encoder = nn.Conv2d(4, self.enc_dim, kernel_size=7, padding=3)
        # self.encoder = nn.Conv2d(6, self.enc_dim, kernel_size=7, padding=3)
        self.encoder_nonlinearity = nn.ReLU()
        self.separator = Separator(self.enc_dim, self.enc_dim, self.bottleneck_dim, self.hidden_dim, self.nhead, self.num_layer)
        self.nonlinearity = nn.ReLU()
        self.decoder = nn.Conv2d(self.enc_dim, 2, kernel_size=1)

    def pad_signal(self, input):
        #input size: (Batch, Time) or (Batch, 1, Time)

        if input.dim() not in [2, 3]:
            raise RuntimeError('Input can only be 2 or 3 dimensional.')

        if input.dim() == 2:
            input = input.unsqueeze(1)

        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win_size - (self.hop_size + nsample % self.win_size) % self.win_size
        if rest > 0:
            pad = torch.zeros(batch_size, 1, rest).type(input.type()).to(input.device)
            input = torch.cat([input, pad], 2)

        pad_extra = torch.zeros(batch_size, 1, self.hop_size).type(input.type()).to(input.device)
        input = torch.cat([pad_extra, input, pad_extra], 2)
        return input, rest

    def init_kernels(self, inverse=False):
        if self.win_type is None:
            window = np.ones(self.win_len)
        else:
            window = get_window(self.win_type, self.win_size, fftbins=True)

        fourier_basis = np.fft.rfft(np.eye(self.fft_len))[: self.win_size]
        real_kernel = np.real(fourier_basis)
        imag_kernel = np.imag(fourier_basis)
        kernel = np.concatenate([real_kernel, imag_kernel], 1).T

        if inverse:
            kernel = np.linalg.pinv(kernel).T

        kernel = kernel * window
        kernel = kernel[:, None, :]
        return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None, :, None].astype(np.float32))

    def ConvSTFT(self, input):
        kernel, _ = self.init_kernels()
        output = F.conv1d(input, kernel.to(input.device), stride=self.hop_size)

        return output

    def ConviSTFT(self, input):
        kernel, window = self.init_kernels(inverse=True)
        output = F.conv_transpose1d(input, kernel.to(input.device), stride=self.hop_size)

        t = window.repeat(1, 1, input.size(-1)) ** 2
        enframe = torch.eye(self.win_size).unsqueeze(1)
        coff = F.conv_transpose1d(t, enframe, stride=self.hop_size)
        output = output / (coff.to(input.device) + 1e-8)

        return output

    def FeaCompression(self, input, factor=0.5):
        input_change = input.float()
        complex_spectrum = torch.complex(input_change[:, 0, :, :], input_change[:, 1, :, :])
        magnitude = torch.abs(complex_spectrum).unsqueeze(1) ** factor
        phase = torch.angle(complex_spectrum).unsqueeze(1)

        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        output = torch.cat((real, imag), dim=1)

        return output

    def FeaDecompression(self, input, factor=0.5):
        input_change = input.float()
        complex_spectrum = torch.complex(input_change[:, 0, :, :], input_change[:, 1, :, :])
        magnitude = torch.abs(complex_spectrum).unsqueeze(1) ** (1 / factor)
        phase = torch.angle(complex_spectrum).unsqueeze(1)

        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        output = torch.cat((real, imag), dim=1)

        return output

    def ComputeSimilarity(self, input, enrollment):
        att = enrollment.transpose(-2, -1) @ input
        att = self.softmax(att)
        output = enrollment @ att

        return output.unsqueeze(0).unsqueeze(0)

    ## matched
    def forward(self, input, seout, enrollment):
        if enrollment.dim() == 2:
            enrollment = enrollment.unsqueeze(1)
        if input.dim() == 2:
            input = input.unsqueeze(1)
        if seout.dim() == 2:
            seout = seout.unsqueeze(1)

        output, rest = self.pad_signal(input)
        seoutput, serest = self.pad_signal(seout)

        batch_size = output.size(0)

        output= self.ConvSTFT(output).contiguous().view(batch_size, 2, self.fft_len // 2 + 1, -1)
        output = self.FeaCompression(output)
        
        seoutput= self.ConvSTFT(seoutput).contiguous().view(batch_size, 2, self.fft_len // 2 + 1, -1)
        seoutput = self.FeaCompression(seoutput)

        similarity = []
        for i in range(batch_size):
            temp, _ = self.pad_signal(enrollment[i])
            temp = self.ConvSTFT(temp).contiguous().view(1, 2, self.fft_len // 2 + 1, -1)
            temp = self.FeaCompression(temp)
            similarity.append(torch.cat([self.ComputeSimilarity(seoutput[i, 0, ...], temp[0, 0, ...]), self.ComputeSimilarity(seoutput[i, 1, ...], temp[0, 1, ...])], dim=1))
        similarity = torch.cat(similarity, dim=0)

        enc_output = self.encoder_nonlinearity(self.encoder(torch.cat((output, similarity), dim=1)))
        masks = self.nonlinearity(self.separator(enc_output).view(batch_size, 1, self.enc_dim, self.fft_len // 2 + 1, -1))
        masked_output = enc_output.unsqueeze(1) * masks
        dec_output = self.decoder(masked_output.view(batch_size, self.enc_dim, self.fft_len // 2 + 1, -1))
        output = self.FeaDecompression(dec_output)
        output = self.ConviSTFT(output.contiguous().view(batch_size, self.fft_len + 2, -1))
        output = output[:, :, self.hop_size:-(rest + self.hop_size)].contiguous()
        output = output.view(batch_size, 1, -1)
        output = output.squeeze(1)
        return output


if __name__ == '__main__':
    torch.manual_seed(0)
    input = torch.randn(1, 1, 8000)
    enrollment = torch.randn(1, 1, 8000)
    se_out = torch.randn(1, 1, 8000)
    nnet = FreqNet()
    output = nnet(input, se_out, enrollment)

    input_b2 = torch.randn(2, 1, 8000)
    enrollment1 = torch.randn(1, 5000)
    enrollment2 = torch.randn(1, 9000)
    enrollment_b2 = [enrollment1, enrollment2]
    output_b2 = nnet(input_b2, enrollment_b2)

    print("{:.3f} million".format(sum([param.nelement() for param in nnet.parameters()]) / 1e6))
    from thop import profile
    macs, params = profile(nnet, inputs=(input, enrollment))
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs / 1e9))
    print('Finish this snippet!')
