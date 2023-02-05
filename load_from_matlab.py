from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import torch

#load omri's results
analog_ppm_empirical_map_GaussianSDR = loadmat('.\\omris_graphs\\GaussianSDR_Graph\\analog_ppm_empirical_map.mat')
analog_ppm_empirical_map_GaussianSDR = analog_ppm_empirical_map_GaussianSDR['y4']
analog_ppm_lower_bound_GaussianSDR = loadmat('.\\omris_graphs\\GaussianSDR_Graph\\analog_ppm_lower_bound.mat')
analog_ppm_lower_bound_GaussianSDR = analog_ppm_lower_bound_GaussianSDR['y1']
burnashev_GaussianSDR = loadmat('.\\omris_graphs\\GaussianSDR_Graph\\burnashev.mat')
burnashev_GaussianSDR = burnashev_GaussianSDR['y3']
sevnic_tuncel_GaussianSDR = loadmat('.\\omris_graphs\\GaussianSDR_Graph\\sevnic_tuncel.mat')
sevnic_tuncel_GaussianSDR = sevnic_tuncel_GaussianSDR['y2']

# load learning net results
net_distortion = torch.load('.\\models\\20_1_results\\distortion.pt')
net_ppm_GaussianSDR = 10 * torch.log10(1/net_distortion)
net_ppm_GaussianSDR = net_ppm_GaussianSDR.detach().numpy()

x = np.arange(4, 16)
plt.plot(x, net_ppm_GaussianSDR, label="net_analog_ppm_GaussianSDR", marker='*', linestyle = 'dashed')
plt.plot(x, np.squeeze(analog_ppm_empirical_map_GaussianSDR)[:12], label="analog_ppm_empirical_map_GaussianSDR", marker='o', linestyle = 'dashed')
plt.plot(x, np.squeeze(analog_ppm_lower_bound_GaussianSDR)[:12], label="analog_ppm_lower_bound_GaussianSDR", linestyle = 'dotted')
plt.plot(x, np.squeeze(burnashev_GaussianSDR)[:12], label="burnashev_GaussianSDR", marker='v', linestyle = 'dashed')
plt.plot(x, np.squeeze(sevnic_tuncel_GaussianSDR)[:12], label="sevnic_tuncel_GaussianSDR", marker='s', linestyle = 'dashed')
plt.legend()
plt.grid()
plt.show()

print('done')

