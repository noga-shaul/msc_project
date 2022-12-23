import torch


def rect(x, E=torch.tensor(1), delta=torch.tensor(1), beta=torch.tensor(10), dt=torch.tensor(0.001), overload=torch.tensor(6.4), noise=True, std=torch.tensor(1)):
    # input: x is tensor scalar in range -0.5<=x<=0.5 represent the time_shift/delta of the rectangular
    #       E is tensor scalar represent the energy limitation
    #       delta is tensor scalar
    #       beta is tensor scalar
    # output: rectang: rectangule with shift of x*delta, and energy E. width=delta/beta, hight=sqrt(E*beta/delta),
    #           tensor size is delta*(1+1/beta)/res+1


    #start = -delta * (1+1/beta) / 2
    #stop = delta * (1+1/beta) / 2
    #steps = 10000 #need to be competable with fc4 size in ppm_net. (stop-start) / res + 1
    #t = torch.linspace(start, stop, steps)
    t = torch.arange(-overload, overload, dt)  #size of ceil(2*overload/dt)
    #start_rect = -delta / (2 * beta) + x * delta
    #stop_rect = delta / (2 * beta) + x * delta
    #rectang = ((start_rect <= t).float()) * ((t <= stop_rect).float()) * torch.sqrt(beta/delta) * torch.sqrt(E)
    x[torch.abs(x) > overload] = overload * torch.sign(x[torch.abs(x) > overload])
    TxPulse = (torch.abs(t-x) < 1 / (2 * beta)).float() * torch.sqrt(beta)
    TxPulse = torch.sqrt(E) * TxPulse / (torch.sum((TxPulse ** 2), 1, keepdim=True) * dt)

    #TxPulse = (np.abs(t - S) < 1 / (2 * beta)) * np.sqrt(beta)
    #TxPulse = np.sqrt(ENRlin) * TxPulse / np.sum((TxPulse ** 2) * dt)

    if noise:
        #std = torch.sqrt(N0/2)
        #n = torch.exp(-0.5*torch.square(t/std))
        noise = std * torch.randn_like(TxPulse)
        noise = torch.sqrt(1 / (2 * dt)) * noise
        TxPulse += noise

    return TxPulse, t





