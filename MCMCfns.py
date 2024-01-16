import numpy as np



def loglikely_2(v, av, sl, **kwargs):

    # v = p[:int(len(p)/2)]
    # av = p[int(len(p)/2):]
    # av = np.tile(av, len(sl.stars)).reshape(len(sl.stars), -1)

    signal = sl.signals
    sigma = sl.signal_errs

    # print('loglikely av shape' ,av.shape)
    val = - 0.5 * np.nansum((signal - sl.model_signals(v, dAVdd = av))**2 / (sigma**2)) # IS THIS WRONG
    if np.isnan(val):
        # print('fail loglikely')
        return -np.inf
    else:
        return val
    # return - 0.5 * np.sum((signal - sl.model_signals(v, dAVdd = av))**2 / (sigma**2)) 

# def logprior_v(v, v_max = 5, prior_mult = 1, **kwargs):
#     if (np.any(np.abs(v) > prior_mult * v_max)):
#         # print('logprior v -inf')
#         return -np.inf
#     return 0.0


def logprior_v(v, v_max = 5, prior_mult = 1, **kwargs):
    if (np.any(v < -8.5)) or (np.any(v > 17.5)):
        # print('logprior v -inf')
        return -np.inf
    return 0.0

def logprior_davdd(av, AV_base = 5, AV_max = 10):   
    # if (np.any(np.abs(av - AV_base) > AV_max)):
    #     # print('av -inf')
    #     return -np.inf
    if ((np.any(av < 0))):
        # print('logprior av -inf')
        return -np.inf
    return 0.0

def logprior_davdd_reg(av,sl, mask = None, **kwargs):
    # print(av.shape)
    # av = np.tile(av, len(sl.stars)).reshape(len(sl.stars), -1) # FOR NOW 
    av = np.copy(av)


    mask = sl.dAVdd_mask
    # mask = av == 0
    av[mask] = np.nan

    # avmed = np.nanmedian(av, axis = 0)
    # avstd = np.nanstd(av, ddof = 1,  axis = 0)
    # avstd[np.isnan(avstd)] = 0.2

    avmed = sl.voxel_dAVdd
    # print(avmed.shape)
    avstd = sl.voxel_dAVdd_std * 10 # should be 10
    # print(avstd.shape)

    # print(av.shape)
    # return 0.0
    # lp_val = np.nansum(np.log(1/(np.sqrt(2 * np.pi) * avstd))) - 0.5 * np.nansum((av - avmed[:, np.newaxis])**2 / (2 * avstd[:, np.newaxis]**2))# first part might not be needed
    # lp_val = np.nansum(- 0.5 * np.nansum((av - avmed[np.newaxis, :])**2 / (2 * avstd[np.newaxis, :]**2)))# first part might not be needed
    lp_val = -np.nansum(np.log(np.sqrt(2 * np.pi))) + np.nansum(- 0.5 * np.nansum((av - avmed)**2 / (2 * avstd**2)))# first part might not be needed

    
    return lp_val
    # return np.sum(np.log(1/(avstd[:, np.newaxis] * np.sqrt(2 * np.pi ))) - 0.5 * (av - avmed[:, np.newaxis])**2 / (2 * avstd[:, np.newaxis]**2)) # first part might not be needed

def logprior_davdd_reg_group(av,sl, mask = None,  width_factor = 3, **kwargs):
    av = np.copy(av)
    mask = sl.dAVdd_mask
    av[mask] = np.nan
    avmed = np.nanmedian(av, axis = 0,)
    avstd = sl.voxel_dAVdd_std


    lp_val = - np.nansum(np.log(np.sqrt(2 * np.pi))) + np.nansum(- 0.5 * np.nansum((av - avmed)**2 / (2 * (width_factor * avstd)**2)))# first part might not be needed
    return lp_val


def logprob_2(p, sl, logprior = logprior_v, loglikely = loglikely_2, **kwargs):
    ndim = len(sl.voxel_dAVdd)
    v = p[ :ndim]
    av = p[ndim:].reshape(-1, ndim)

    # print(av.shape)

    lp = logprior(v, **kwargs)
    lp_davdd = logprior_davdd(av, AV_base = sl.dAVdd)
    lp_davdd_reg = logprior_davdd_reg(av, sl, **kwargs)
    lp_davdd_reg_group = logprior_davdd_reg_group(av, sl)

    if (not np.isfinite(lp)) | (not np.isfinite(lp_davdd)) | (not np.isfinite(lp_davdd_reg)):
        # print('fail logprob')
        return -np.inf
    return lp + lp_davdd + lp_davdd_reg +  loglikely_2(v, av, sl = sl, **kwargs) + lp_davdd_reg_group # group term added 10.13

def logprob_avfix(p,sl, av = None,  logprior = logprior_v, loglikely = loglikely_2, **kwargs):
    ndim = len(sl.voxel_dAVdd)
    v = p[:ndim]

    # av = av.reshape(-1, ndim)

    lp = logprior(v, **kwargs)
    if (not np.isfinite(lp)):
        return -np.inf
    return lp + loglikely_2(v, av, sl = sl, **kwargs)

