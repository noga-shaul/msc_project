import torch


def rect(x, E=torch.tensor(1), delta=torch.tensor(1), beta=torch.tensor(10), res=torch.tensor(0.001), noise=True, std=torch.tensor(7)):
    # input: x is tensor scalar in range -0.5<=x<=0.5 represent the time_shift/delta of the rectangular
    #       E is tensor scalar represent the energy limitation
    #       delta is tensor scalar
    #       beta is tensor scalar
    # output: rectang: rectangule with shift of x*delta, and energy E. width=delta/beta, hight=sqrt(E*beta/delta),
    #           tensor size is delta*(1+1/beta)/res+1

    res = 0.001
    start = -delta * (1+1/beta) / 2
    stop = delta * (1+1/beta) / 2
    steps = 10000 #need to be competable with fc4 size in ppm_net. (stop-start) / res + 1
    t = torch.linspace(start, stop, steps)
    start_rect = -delta / (2 * beta) + x * delta
    stop_rect = delta / (2 * beta) + x * delta
    rectang = ((start_rect <= t).float()) * ((t <= stop_rect).float()) * torch.sqrt(beta/delta) * torch.sqrt(E)

    if noise:
        #std = torch.sqrt(N0/2)
        #n = torch.exp(-0.5*torch.square(t/std))
        n = std * torch.randn_like(rectang)
        rectang += n

    return rectang





