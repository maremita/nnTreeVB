import torch

__author__ = "amine remita"

__all__ = [
        "pruning",
        "pruning_known_ancestors"
        ]

# EvoTreeVGTRW_KL
def pruning(arbre, x, tm, pi):

    #print("x shape {}".format(x.shape))
    # [sample_size, n_dim, m_dim, x_dim]
    #print(x)
    #
    #print("pi")
    #print("pi shape {}".format(pi.shape))
    # [sample_size, x_dim]
    #print(pi)
    #
    #print("tm")
    #print("tm shape {}".format(tm.shape))
    # [sample_size, b_dim, x_dim, x_dim]
    #print(tm)

    # Algorithm from Mol Evolution Book (Yang) page 105
    # Felsenstein (Pruning) algorithm
    # The implementation inspired from VBPI:
    # https://github.com/zcrabbit/vbpi-nf/blob/3dead78900da64c9634fece931f201011ce51aa1/code/phyloModel.py#L50
    scaler_list = []

    for node in arbre.traverse("postorder"):
        if node.is_leaf():
            #print("shape x {}".format(x.shape))
            node.state = x[:, :,node.rank,:].detach() 
            # x [sample_size, n_dim, b_dim, x_dim]
            #print("leaf {}\t{}".format(node.name, node.state.shape)) 
            # [sample_size, n_dim, x_dim]
        else:
            node.state = 1.0
            #print("\nNode {}".format(node.name))

            for child in node.children:
                #print("Child {} {}".format(child.name, child.rank))
                #print("tm[:, {}].shape {}".format(child.rank, 
                #    tm[:, child.rank].shape))
                # [sample_size, x_dim, x_dim]
                
                #print("node.state.shape {}".format(node.state.shape))
                # [sample_size, n_dim, x_dim]

                parlial_ll = torch.einsum("bcij,bjk->bcik",
                        (child.state.unsqueeze(-2),
                            tm[:, child.rank]))\
                                    .squeeze(-2).clamp(min=0., max=1.)
                #print("parlial_ll {}".format(parlial_ll.shape)) 
                # [sample_size, n_dim, x_dim]

                node.state *= parlial_ll
                #print("node {}\t{}".format(node.name,
                #    node.state.shape)) 
                # [sample_size, n_dim, x_dim]

            scaler = torch.sum(node.state, -1).unsqueeze(-1)
            #print("scaler shape {}".format(scaler.shape))
            # [9, 7, 1]
            node.state /= scaler
            scaler_list.append(scaler)
            #print()

            #print("\npi shape {}".format(pi.shape))
            # [sample_size, 1, x_dim]
            
            #  print(pi)        

            #  print("\nroot shape {}".format( arbre.state.shape))
            # [sample_size, n_dim, x_dim, 1]
            #  print(arbre.state)

            #b  i  j            b   c   j   k       
            #  pi unseqz(-2)   : [sample_size, 1, x_dim]  root unseqz(-1) : [sample_size, n_dim, x_dim, 1]

            #logl = torch.einsum("bij,bcjk->bcik", (pi.unsqueeze(-2),
            #    arbre.state.unsqueeze(-1))).log().mean(0).flatten()

    scaler_list.append(torch.einsum("bij,bcjk->bcik",
        (pi.unsqueeze(-2), arbre.state.unsqueeze(-1))).squeeze(-1))
    #logl = torch.sum(torch.log(torch.stack(scaler_list)))
    #stack = torch.stack(scaler_list)
    #print("\nstack {}".format(stack.shape))
    # [nb_scaler, sample, n_dim, 1]

    logl = torch.sum(torch.log(torch.stack(scaler_list)),
            dim=0).mean(0).flatten()
    #print("\nlogl")
    #print(logl.shape) #[n_dim]
    #print(logl)

    return logl

def pruning_known_ancestors(arbre, x, a, tm, pi):

        # Assign each node its sites
        for node in arbre.traverse("postorder"):
            if node.is_leaf():
                node.sites = x[:, :, node.rank, :] #.detach() # [sample_size, n_dim, x_dim]
            else:
                node.sites = a[:, :, node.ancestral_rank, :]
                # node.sites = 1.0

        log_pi_a = torch.einsum("bij,bcjk->bcik", (pi.unsqueeze(-2), arbre.sites.unsqueeze(-1))).log().squeeze(-1).squeeze(-1)
        #         print("\nlog_pi_a")
        #         print(log_pi_a.shape)  # [sample_size, n_dim, 1]
        #         print(log_pi_a)

        #         ll_cumul = torch.zeros_like(log_pi_a)
        ll_cumul = 0.0
        for node in arbre.traverse("postorder"):
            if not node.is_leaf():
                # Internal node
                #log_pi_node = torch.einsum("bij,bcjk->bcik", (pi.unsqueeze(-2), node.sites.unsqueeze(-1))).log().view(x.shape[:2])
                #ll_cumul += log_pi_node

                #print("\nlog_pi_node")
                #print(log_pi_node.shape)  # [sample_size, n_dim, 1]
                #print(log_pi_node)

                for child in node.children:
                    #print("Child {}".format(child.name))
                    #print("tm[:, {}].shape {}".format(child.rank, tm[:, child.rank].shape)) # [sample_size, x_dim, x_dim]
                    #print("child.sites.shape {}".format(child.sites.shape)) # [sample_size, n_dim, x_dim]
                    
                    parlial_ll = torch.einsum("bcij,bjk->bcik", (child.sites.unsqueeze(-2), tm[:, child.rank])).squeeze(-2).clamp(min=0., max=1.)
                    #print("parlial_ll {}".format(parlial_ll.shape))  # [sample_size, n_dim, x_dim]
                    #print(parlial_ll)
                    #print()
                    #print("node.sites.shape {}".format(node.sites.shape)) # [sample_size, n_dim, x_dim]
                    #print(node.sites)
                     
                    #ll_cumul += (parlial_ll.squeeze().dot(node.sites.squeeze())).log()
                    ll_cumul += torch.log(torch.einsum("bij,bij->bi", (parlial_ll, node.sites))) #.log
                    #print("ll_cumul {}".format(ll_cumul.shape))  # [sample_size, n_dim, x_dim]

        logl = ll_cumul + log_pi_a
        #print("logl {}".format(logl.shape))

        return logl
