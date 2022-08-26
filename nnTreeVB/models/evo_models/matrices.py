import torch

__author__ = "Amine Remita"


# Adpated from https://github.com/zcrabbit/vbpi-nf/blob/main/code/rateMatrix.py#L50
def build_GTR_matrix(rates, pden):

    # print("rates {}".format(rates.shape)) # [sample_size, r_dim]
    # print("pden {}".format(pden.shape)) # [sample_size, f_dim]
    #print(pden)
    
    sample_size = rates.shape[0]

    pA = pden[...,0]
    pG = pden[...,1]
    pC = pden[...,2]
    pT = pden[...,3]

    AG = rates[...,0]
    AC = rates[...,1]
    AT = rates[...,2]
    GC = rates[...,3]
    GT = rates[...,4]
    CT = rates[...,5]
    
    # print("pA: {}".format(pA))
    # print("pG: {}".format(pG))
    # print("pC: {}".format(pC))
    # print("pT: {}".format(pT))
    # print()

    # print("AG: {}".format(AG))
    # print("AC: {}".format(AC))
    # print("AT: {}".format(AT))
    # print("GC: {}".format(GC))
    # print("GT: {}".format(GT))
    # print("CT: {}".format(CT))

    rate_matrix = torch.zeros((sample_size, 4, 4))

    for i in range(4):
        for j in range(4):
            if j!=i:
                rate_matrix[..., i,j] = pden[...,j]
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
    # substitution ate is 1. Time will be the distance : t = d/1
    # and measured as susbtitution per site
    # See page 30 in MESA Book (Yang 2014)
    rate_matrix = torch.einsum("b,bij->bij", (beta, rate_matrix))
    # print("\nrate_matrix * beta")
    # print(rate_matrix.shape) # [sample_size, x_dim, x_dim]
    # print(rate_matrix)
 
    return rate_matrix

def build_GTR_transition_matrix(t, r, pi):
    # print("t shape {}".format(t.shape)) # [sample_size]
    # print(t)

    rateM = build_GTR_matrix(r, pi)
    # print("rateM shape {}".format(rateM.shape)) #[sample_size, x_dim, x_dim]
    # print(rateM)

    tm = torch.matrix_exp(
            torch.einsum("bij,bck->bcij", (rateM, t))).clamp(
                    min=0.0, max=1.0)

    # print("\ntm sahpe {}".format(tm.shape)) # [sample_size, x_dim, x_dim]
    #print(tm)

    return tm
