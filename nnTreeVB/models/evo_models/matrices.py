import torch

__author__ = "Amine Remita"

# Adpated from https://github.com/zcrabbit/vbpi-nf/blob/main/code/rateMatrix.py#L50

# #############
# JC69 matrices
# #############
def build_JC69_matrix(rate=1.0/3):

    rate_matrix = rate * torch.ones((4,4))

    for i in range(4):
        rate_matrix[i,i] = -1.0

    return rate_matrix

def build_JC69_transition_matrix(b=1.):

    rateM = build_JC69_matrix()
    # print("rateM shape {}".format(rateM.shape)) #[x_dim, x_dim]
    # print(rateM)

    tm = torch.matrix_exp(
            torch.einsum("ij,bck->bcij", (rateM, b))).clamp(
                    min=0.0, max=1.0)

    # print("\ntm sahpe {}".format(tm.shape)) # [sample_size, x_dim, x_dim]
    #print(tm)

    return tm


# #############
# K80 matrices
# #############
def build_K80_matrix(kappa):
    #print("kappa shape {}".format(kappa.shape))
    #[sample_size, 1]
    sample_size = kappa.shape[0]
    freqs = torch.ones(4)/4
    pA, pG, pC, pT = freqs

    rate_matrix = torch.zeros((sample_size, 4, 4))
    
    #print(kappa[...,0])

    for i in range(4):
        for j in range(4):
            if j!=i:
                rate_matrix[..., i,j] = freqs[..., j]
            if i+j == 1 or i+j == 5:
                rate_matrix[..., i,j] *= kappa[..., 0]

    for i in range(4):
        rate_matrix[..., i,i] = - rate_matrix.sum(dim=-1)[..., i]
 
    # Scaling factor
    beta = 1.0/(2*(pA+pG)*(pC+pT) + 2*kappa.flatten()*(pA*pG+pC*pT))
    #print("beta shape {}".format(beta.shape))
    #[sample_size]

    # Multiply the rate matrix by beta so that the average 
    # substitution ate is 1. Time will be the distance : b = d/1
    # and measured as susbtitution per site
    # See page 30 in MESA Book (Yang 2014)
    rate_matrix = torch.einsum("b,bij->bij", (beta, rate_matrix))

    return rate_matrix


def build_K80_transition_matrix(b=1., kappa=1.):
    rateM = build_K80_matrix(kappa)
    #print("rateM shape {}".format(rateM.shape))
    #[sample_size, x_dim, x_dim]
    # print(rateM)

    tm = torch.matrix_exp(
            torch.einsum("bij,bck->bcij", (rateM, b))).clamp(
                    min=0.0, max=1.0)

    #print("\ntm sahpe {}".format(tm.shape))
    # [sample_size, x_dim, x_dim]
    #print(tm)

    return tm


# #############
# HKY matrices
# #############
def build_HKY_matrix(freqs, kappa):
    #print("kappa shape {}".format(kappa.shape))
    #[sample_size, 1]

    sample_size = kappa.shape[0]

    pA = freqs[...,0]
    pG = freqs[...,1]
    pC = freqs[...,2]
    pT = freqs[...,3]
    #print("pA shape {}".format(pA.shape))
    # [sample_size]

    rate_matrix = torch.zeros((sample_size, 4, 4))

    for i in range(4):
        for j in range(4):
            if j!=i:
                rate_matrix[..., i,j] = freqs[..., j]
            if i+j == 1 or i+j == 5:
                rate_matrix[..., i,j] *= kappa[..., 0]

    for i in range(4):
        rate_matrix[..., i,i] = - rate_matrix.sum(dim=-1)[..., i]
 
    # Scaling factor
    beta = 1.0/(2*(pA+pG)*(pC+pT) + 2*kappa.flatten()*(pA*pG+pC*pT))
    #print("beta shape {}".format(beta.shape))
    #[sample_size]

    # Multiply the rate matrix by beta so that the average 
    # substitution ate is 1. Time will be the distance : b = d/1
    # and measured as susbtitution per site
    # See page 30 in MESA Book (Yang 2014)
    rate_matrix = torch.einsum("b,bij->bij", (beta, rate_matrix))

    return rate_matrix


def build_HKY_transition_matrix(b=1., freqs=0.25, kappa=1.):
    rateM = build_HKY_matrix(freqs, kappa)
    #print("rateM shape {}".format(rateM.shape))
    #[sample_size, x_dim, x_dim]
    # print(rateM)

    tm = torch.matrix_exp(
            torch.einsum("bij,bck->bcij", (rateM, b))).clamp(
                    min=0.0, max=1.0)

    #print("\ntm sahpe {}".format(tm.shape))
    # [sample_size, x_dim, x_dim]
    #print(tm)

    return tm


# #############
# GTR matrices
# #############
def build_GTR_matrix(rates, freqs):

    # print("rates {}".format(rates.shape)) # [sample_size, r_dim]
    # print("freqs {}".format(freqs.shape)) # [sample_size, f_dim]
    #print(freqs)

    sample_size = rates.shape[0]

    pA = freqs[...,0]
    pG = freqs[...,1]
    pC = freqs[...,2]
    pT = freqs[...,3]

    AG = rates[...,0]
    AC = rates[...,1]
    AT = rates[...,2]
    GC = rates[...,3]
    GT = rates[...,4]
    CT = rates[...,5]
 
    rate_matrix = torch.zeros((sample_size, 4, 4))

    for i in range(4):
        for j in range(4):
            if j!=i:
                rate_matrix[..., i,j] = freqs[...,j]
                if i+j == 1:
                    rate_matrix[..., i,j] *= AG
                if i+j == 2:
                    rate_matrix[..., i,j] *= AC
                if i+j == 3 and abs(i-j) > 1:
                    rate_matrix[..., i,j] *= AT
                if i+j == 3 and abs(i-j) == 1:
                    rate_matrix[..., i,j] *= GC
                if i+j == 4:
                    rate_matrix[..., i,j] *= GT
                if i+j == 5:
                    rate_matrix[..., i,j] *= CT

    for i in range(4):
        rate_matrix[..., i,i] = - rate_matrix.sum(dim=-1)[..., i]

    # Scaling factor
    beta = (1.0/(
        2*(AG*pA*pG+AC*pA*pC+AT*pA*pT+GC*pG*pC+GT*pG*pT+CT*pC*pT)))
    # print("\nbeta")
    # print(beta.shape) #[sample_size]
    # print(beta)

    # Multiply the rate matrix by beta so that the average 
    # substitution ate is 1. Time will be the distance : b = d/1
    # and measured as susbtitution per site
    # See page 30 in MESA Book (Yang 2014)
    rate_matrix = torch.einsum("b,bij->bij", (beta, rate_matrix))
    # print("\nrate_matrix * beta")
    # print(rate_matrix.shape) # [sample_size, x_dim, x_dim]
    # print(rate_matrix)
 
    return rate_matrix

def build_GTR_transition_matrix(b, rates=0.16, freqs=0.25):
    # print("b shape {}".format(b.shape)) # [sample_size]
    # print(b)

    rateM = build_GTR_matrix(rates, freqs)
    # print("rateM shape {}".format(rateM.shape)) #[sample_size, x_dim, x_dim]
    # print(rateM)

    tm = torch.matrix_exp(
            torch.einsum("bij,bck->bcij", (rateM, b))).clamp(
                    min=0.0, max=1.0)

    # print("\ntm sahpe {}".format(tm.shape)) # [sample_size, x_dim, x_dim]
    #print(tm)

    return tm

def build_transition_matrix(subs_model, args):

    if subs_model == "jc69":
        # args = {b}
        tm = build_JC69_transition_matrix(**args)

    elif subs_model == "k80":
        # args = {b, kappa}
        tm = build_K80_transition_matrix(**args)

    elif subs_model == "hky":
        # args ={b, freqs, kappa}
        tm = build_HKY_transition_matrix(**args)

    elif subs_model == "gtr":
        # args = {b, rates, freqs}
        tm = build_GTR_transition_matrix(**args)

    else:
        raise ValueError("Substitution model key {} is not valid.\n"\
                "Valid values : jc69 | k80 | hky | gtr".format(
                    subs_model))

    return tm
