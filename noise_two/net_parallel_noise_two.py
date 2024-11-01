import torch
import torch.nn as nn
import torch.nn.functional as F
#from rect import rect
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    def __init__(self, params, num_joint_samples, num_encoded_outputs):
        super().__init__()
        #self.fc_t = nn.Linear(1,1)
        self.num_joint_samples = num_joint_samples
        base_filters = 16 * num_joint_samples
        #modulation layers
        self.fc1 = nn.Linear(num_joint_samples, base_filters)
        self.fc2 = nn.Linear(base_filters, base_filters*2)
        self.fc3 = nn.Linear(base_filters*2, num_encoded_outputs)
        #demodulation layers
        self.fc1_de = nn.Linear(num_encoded_outputs, base_filters)
        self.fc2_de = nn.Linear(base_filters, base_filters*2)
        self.fc3_de = nn.Linear(base_filters*2, num_joint_samples)
        #self.conv1 = nn.Conv1d(1, 16, 1, padding=1)
        #self.pool = nn.MaxPool1d(2)
        #self.conv2 = nn.Conv1d(16, 32, 1, padding=1)
        #self.conv3 = nn.Conv1d(32, 64, 1, padding=1)
        #self.fc4 = nn.Linear(64, 1)  # input size 32*steps/4

        self.params = {}
        for param_name, param_val in params.items():
            #print(param_name, ":", param_val)
            self.params[param_name] = torch.tensor(param_val).clone().detach()

    def to(self, device):
        new_self = super().to(device)
        for param_name, param_val in self.params.items():
            new_self.params[param_name] = self.params[param_name].to(device)
        new_self.device = device
        return new_self


    #@torch.no_grad()

    def rect(self, x, E, beta, dt, overload, noise_std, noise=True, delta=torch.tensor(1)):
        # input: x is tensor scalar in range -0.5<=x<=0.5 represent the time_shift/delta of the rectangular
        #       E is tensor scalar represent the energy limitation
        #       delta is tensor scalar
        #       beta is tensor scalar
        # output: rectang: rectangule with shift of x*delta, and energy E. width=delta/beta, hight=sqrt(E*beta/delta),
        #           tensor size is delta*(1+1/beta)/res+1

        #beta = beta.to(self.device)
        #dt = dt.to(self.device)
        #overload = overload.to(self.device)
        #noise_std = noise_std.to(self.device)

        # start = -delta * (1+1/beta) / 2
        # stop = delta * (1+1/beta) / 2
        # steps = 10000 #need to be competable with fc4 size in ppm_net. (stop-start) / res + 1
        # t = torch.linspace(start, stop, steps)
        start_t = torch.squeeze(-overload-1/(beta))
        end_t = torch.squeeze(overload+1/(beta))
        dt = torch.squeeze(dt)
        t = torch.arange(start_t, end_t+dt, dt, device=self.device)  # size of ceil(2*overload+1/beta)/dt
        #t.to(self.device)
        # start_rect = -delta / (2 * beta) + x * delta
        # stop_rect = delta / (2 * beta) + x * delta
        # rectang = ((start_rect <= t).float()) * ((t <= stop_rect).float()) * torch.sqrt(beta/delta) * torch.sqrt(E)
        x[torch.abs(x) > overload] = overload * torch.sign(x[torch.abs(x) > overload])
        TxPulse = (torch.abs(t - x) < 1 / (2 * beta)).float() * torch.sqrt(beta)
        TxPulse = torch.sqrt(E) * TxPulse / (torch.sum((TxPulse ** 2), 1, keepdim=True) * dt)

        # TxPulse = (np.abs(t - S) < 1 / (2 * beta)) * np.sqrt(beta)
        # TxPulse = np.sqrt(ENRlin) * TxPulse / np.sum((TxPulse ** 2) * dt)

        if noise:
            # std = torch.sqrt(N0/2)
            # n = torch.exp(-0.5*torch.square(t/std))
            #torch.manual_seed(0)

            noise = noise_std * torch.randn_like(TxPulse)
            #print(noise)
            noise = torch.sqrt(1 / (2 * dt)) * noise
            TxPulse += noise

            #for debug:
            TxPulse_zeros = (torch.abs(t - torch.zeros_like(x)) < 1 / (2 * beta)).float() * torch.sqrt(beta)
            TxPulse_zeros = torch.sqrt(E) * TxPulse_zeros / (torch.sum((TxPulse_zeros ** 2), 1, keepdim=True) * dt)
            #noise_shift =
            noise_zeros = torch.roll(noise, -int((x[0] // dt).item()))
            TxPulse_zeros += noise_zeros

        return TxPulse, t, TxPulse_zeros

    @torch.no_grad()
    def find_offset(self, rect_signal, t, x, rect_signal_zeros):
        #convolution with ppm pulse
        conv_kernel = torch.ones(351, device=self.device) * torch.sqrt(self.params['beta'])
        conv_kernel = conv_kernel / torch.sqrt(torch.sum(torch.square(conv_kernel) * self.params['res']))
        conv_kernel = conv_kernel.repeat(1, 1, 1)
        #x = torch.stack((x,t.repeat(x.size(0),1)),1)
        signal_heat_map = F.conv1d(rect_signal, conv_kernel, padding='same')
        #signal_heat_map_zeros = F.conv1d(rect_signal_zeros, conv_kernel, padding='same')
        heat_map_max = torch.max(signal_heat_map,2).values
        #choosing max
        corr_peak_ind = torch.argmax(signal_heat_map, 2)
        #corr_peak_ind = torch.argmax(torch.sqrt(self.params['E']) * signal_heat_map * self.params['res']
        #                             - 0.5 * (t ** 2), 2)
        t = t.repeat(corr_peak_ind.size())
        detect = t[torch.arange(corr_peak_ind.size(0)), torch.squeeze(corr_peak_ind)]
        detect = detect.unsqueeze(1)
        offset = detect - x
        return offset, heat_map_max

    def forward(self, sample, training=True, noise=True):
        x = F.relu(self.fc1(sample))
        x = F.relu(self.fc2(x))
        xs_encoded = self.fc3(x)  # x here is scalar

        out_shift = xs_encoded

        #y = x
        #x = self.fc_t(x)
        #x = x*0+y

        xs_transmitted = torch.zeros_like(xs_encoded)
        for encoded_sample_ind in range(xs_encoded.shape[1]):
            x_encoded = xs_encoded[:, encoded_sample_ind].unsqueeze(1)
            x_encoded=torch.clamp(x_encoded, min=-self.params['overload'], max=self.params['overload'])
            #x_t=x
            #x_t2=x
            if training:
                rect_signal, t, rect_signal_zeros = self.rect(
                    torch.zeros_like(x_encoded),
                    #x_t,
                    E=self.params['E']*self.num_joint_samples,
                    beta=self.params['beta'],
                    dt=self.params['res'],
                    overload=self.params['overload'],
                    noise_std=self.params['std'],
                    noise=noise,
                    delta=self.params['delta'])

                rect_signal = torch.unsqueeze(rect_signal, 1)
                rect_signal_zeros = torch.unsqueeze(rect_signal_zeros, 1)
                #plt.plot(t.cpu().numpy(), np.squeeze(rect_signal[0].cpu().numpy()))
                #plt.show()

                offset, heat_map_max = self.find_offset(rect_signal, t, torch.zeros_like(x_encoded), rect_signal_zeros)
                #offset = torch.normal(torch.zeros_like(x),torch.ones_like(x)*0.2)
                #print(torch.mean(offset), torch.std(offset))
                #print(offset)
                #plt.plot(offset.cpu().numpy()[:, 0]);
                #plt.show()
                #print(torch.mean(offset))

            else:
                rect_signal, t, rect_signal_zeros = self.rect(
                    #torch.zeros_like(x),
                    x_encoded,
                    E=self.params['E']*self.num_joint_samples,
                    beta=self.params['beta'],
                    dt=self.params['res'],
                    overload=self.params['overload'],
                    noise_std=self.params['std'],
                    noise=noise,
                    delta=self.params['delta'])

                rect_signal = torch.unsqueeze(rect_signal, 1)

                # plt.plot(t.cpu().numpy(), np.squeeze(rect_signal[0].cpu().numpy()))
                # plt.show()

                offset, heat_map_max = self.find_offset(rect_signal, t, x_encoded, rect_signal_zeros)

            x_encoded = x_encoded + offset

            x_encoded[x_encoded > t[-1]] = x_encoded[x_encoded > t[-1]] + t[0] - t[-1]
            x_encoded[x_encoded < t[0]] = x_encoded[x_encoded < t[0]] + t[-1] - t[0]
            xs_transmitted[:,encoded_sample_ind] = x_encoded[:,0]



        xs_decoded = (F.relu(self.fc1_de(xs_transmitted))) #if needed self.pool can be add
        xs_decoded = (F.relu(self.fc2_de(xs_decoded)))
        xs_decoded = (self.fc3_de(xs_decoded))

        return xs_decoded, out_shift
