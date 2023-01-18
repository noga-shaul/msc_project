import numpy as np
import matplotlib.pyplot as plt


def compute_upper_bound():
    ENRbound = np.arange(4, 16+0.125, 0.125)
    ENRboundLin = 10 ** (ENRbound / 10)
    ENR_opt = 15
    ENR_opt_lin = 10 ** (ENR_opt / 10)
    beta_opt = (13 / 8) ** (1 / 3) * (ENR_opt_lin) ** (-5 / 6) * np.exp(ENR_opt_lin / 6)
    # exact upper bound

    D_S = ((13 / 8) + np.sqrt(2 / beta_opt) * (np.sqrt(2 * beta_opt * ENRboundLin) - 1) * np.exp( \
            -ENRboundLin * (1 - 1 / np.sqrt(2 * beta_opt * ENRboundLin)) ** 2)) /  \
            ((np.sqrt(beta_opt * ENRboundLin) - 1 / np.sqrt(2)) ** 4) + np.exp(-beta_opt * ENRboundLin) / (beta_opt ** 2)


    D_L = 2 * beta_opt * np.sqrt(ENRboundLin) * np.exp(-ENRboundLin / 2) * ( \
            1 + 3 * np.sqrt(2 * np.pi / ENRboundLin) + 12 * np.exp(-1) / (beta_opt * np.sqrt(ENRboundLin)) \
            + 8 * np.exp(-1) / (np.sqrt(8 * np.pi) * beta_opt) + np.sqrt(8 / (np.pi * ENRboundLin)) + 12 ** (3 / 2) * np.exp(-3 / 2) / ( \
                        beta_opt * np.sqrt(32 * np.pi * ENRboundLin)))

    optDist_Analytic = (D_S + D_L)

    # Asymptotic bound
    D_S_asymp = (13 / 8) / (ENRboundLin * beta_opt) ** 2
    D_L_asymp = 2 * beta_opt * np.sqrt(ENRboundLin) * np.exp(-ENRboundLin / 2)
    optDist_Analytic_asymp = (D_S_asymp + D_L_asymp)

    plt.plot(ENRbound, 10*np.log10(1/optDist_Analytic))
    plt.plot(ENRbound, 10*np.log10(1/optDist_Analytic_asymp))
    plt.show()
    print()


compute_upper_bound()


