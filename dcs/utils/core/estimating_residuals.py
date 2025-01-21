import numpy as np


def estimate_residuals(Yt_stats):
    L, nvar, temp = Yt_stats['OLS']['At'].shape

    bt = np.full((nvar, L), np.nan)
    Sigma_Et = np.full((L, nvar, nvar), np.nan)
    sigma_Et = np.full(L, np.nan)

    for t in range(L):
        Sigma_Xt = np.squeeze(Yt_stats['Sigma'][t, :nvar, :nvar])
        Sigma_Xp = np.squeeze(Yt_stats['Sigma'][t, nvar:, nvar:])
        Sigma_XtXp = np.reshape(np.squeeze(Yt_stats['Sigma'][t, :nvar, nvar:]), (nvar, temp))
        coeff = np.reshape(np.squeeze(Yt_stats['OLS']['At'][t, :, :]), (nvar, temp))

        bt[:, t] = Yt_stats['mean'][:nvar, t] - np.dot(coeff, Yt_stats['mean'][nvar:, t])
        Sigma_Et[t, :, :] = (Sigma_Xt - np.dot(Sigma_XtXp, coeff.T) - np.dot(coeff, Sigma_XtXp.T)
                            + np.dot(np.dot(coeff, Sigma_Xp), coeff.T))
        sigma_Et[t] = np.trace(np.squeeze(Sigma_Et[t, :, :]))

    # Ensure that no negative values exist (if needed)
    # Sigma_Et[Sigma_Et < 0] = 0
    # sigma_Et[sigma_Et < 0] = 0

    return bt, Sigma_Et, sigma_Et
