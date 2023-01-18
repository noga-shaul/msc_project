import numpy as np
from rect import rect
import torch
import matplotlib.pyplot as plt


def fconv(x, Lx, Ly, Ly2, H):
    # Convolution in frequency domain using power of 2 fft Since the input signal is real
    # we use known fft identities to accelerate the fft

    # input fft
    X = np.fft.fft(x, Ly2)

    # multiply with precalculated freq domain signal
    Y = X * H  # check dimensions

    # inverse fft and truncation
    y = np.real(np.fft.ifft(Y, Ly2))
    y = y[:Ly]

    if Lx % 2 == 0:
        y = y[Lx//2:-(Lx//2) + 1]
    else:
        y = y[(Lx-1)//2:-((Lx+1)//2) + 1]

    return y


def simulateGaussianPPM(beta, ENRlin, Nrun, overload, dt):

    #overload = 6.4
    #dt = 1/(350*beta)
    t = np.arange(-overload, overload+dt, dt)
    ppmPulse = (abs(t) < 1/(2*beta)) * np.sqrt(beta)
    ppmPulse = ppmPulse / np.sqrt(np.sum(np.square(ppmPulse)*dt))

    Lx = ppmPulse.shape[0]
    Ly = ppmPulse.shape[0] + ppmPulse.shape[0] - 1
    Ly2 = 2**(Ly - 1).bit_length()  #next pow 2
    PPMfreq = np.fft.fft(ppmPulse, Ly2)
    currMSE_MAP = np.zeros(Nrun)

    for n in range(Nrun):
        #generate source - Gaussian Source with edge truncation
        S = np.random.randn(1)
        #S = 0
        if np.abs(S) > overload:
            S = overload * np.sign(S)

        # create rect with noise:
        r, _ = rect(torch.tensor(S).unsqueeze(0), E=torch.tensor(ENRlin), beta=torch.tensor(beta),
                       dt=torch.tensor(dt), overload=torch.tensor(overload))
        #TxPulse = (np.abs(t - S) < 1 / (2 * beta)) * np.sqrt(beta)
        #TxPulse = np.sqrt(ENRlin) * TxPulse / np.sum((TxPulse**2) * dt)

        #noise = np.random.randn(t.shape[0])
        #noise = np.sqrt(1 / (2 * dt)) * noise
        #r = TxPulse + noise

        # PPM Correlator receiver
        PPMcorr = fconv(r.squeeze(), Lx, Ly, Ly2, PPMfreq)
        #PPMcorr = np.convolve(r, ppmPulse, 'same')

        maxIdx = np.argmax(np.sqrt(ENRlin) * PPMcorr * dt - 0.5 * (t**2))
        sHat_MAP = t[maxIdx]
        currMSE_MAP[n] = (S - sHat_MAP) ** 2


        point = 1

    MSE = np.sum(currMSE_MAP)/Nrun
    return MSE





