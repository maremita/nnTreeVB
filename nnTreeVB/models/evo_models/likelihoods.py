import torch

__author__ = "amine remita"

__all__ = [
        "pruning",
        "pruning_rescaled",
        "pruning_known_ancestors"
        ]

def pruning(arbre, x, tm, pi):

    for node in arbre.traverse("postorder"):
        if node.is_leaf():
            node.state = x[:, :, node.postrank, :]
            # x [sample_size, n_dim, b_dim, x_dim]
            #print("leaf {}\t{}\t{}\t{}".format(node.name,
            #    node.postrank, node.state.shape, node.dist)) 
            # [sample_size, n_dim, x_dim]
        else:
            node.state = 1.0
            #print("\nNode {}".format(node.name))
            #print("node.state.shape {}".format(
            #    node.state.shape))

            for child in node.children:
                #print("Child {} {}".format(child.name,
                #    child.postrank))

                #print(tm[:, child.postrank].shape)
                #print(child.state.unsqueeze(-1).shape)
                node.state *= torch.einsum("bij,bcjk->bcik",
                        tm[:, child.postrank],
                        child.state.unsqueeze(-1)).squeeze(
                                -1).clamp(min=0., max=1.)

                #FIXME: Another alternative to compute partials
                # Does not work correctly when sample_size>1
                #node.state *= (
                #        tm[:, child.postrank].unsqueeze(-3) @
                #        child.state.unsqueeze(-1)).squeeze(
                #                -1).clamp(min=0., max=1.)

            #print("node {}\t{}\t{}\t{}".format(node.name,
            #    node.postrank, node.state.shape, node.dist))
            #print(node.postrank,
            #        node.state.sum(-1).log().sum(),
            #        node.state.shape)

    #print(arbre.state.shape)
    #print(pi.shape)
    #logl = torch.log(torch.einsum("bij,bcjk->bcik",
    #        (pi.unsqueeze(-2), 
    #            arbre.state.unsqueeze(-1)))).squeeze(
    #                    -1).squeeze(-1)

    logl = torch.log(torch.sum(torch.einsum("bj,bcj->bcj", 
        (pi, arbre.state)), -1))

    #logl = torch.log(pi @ arbre.state.unsqueeze(-1)).squeeze(
    #        -1).squeeze(-1)

    #print(logl.shape)
    #print(logl)

    return logl


def pruning_rescaled(arbre, x, tm, pi):

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

    # See https://stromtutorial.github.io/linux/steps/step-12/00-the-large-tree-problem.html

    # The implementation inspired from VBPI:
    # https://github.com/zcrabbit/vbpi-nf/blob/3dead78900da64c9634fece931f201011ce51aa1/code/phyloModel.py#L50
    scaler_list = []

    for node in arbre.traverse("postorder"):
        if node.is_leaf():
            #print("shape x {}".format(x.shape))
            node.state = x[:, :, node.postrank, :]
            # x [sample_size, n_dim, b_dim, x_dim]
            #print("leaf {}\t{}".format(node.name,
            #    node.state.shape)) 
            # [sample_size, n_dim, x_dim]
        else:
            node.state = 1.0
            #print("\nNode {}".format(node.name))

            for child in node.children:
                #print("Child {} {}".format(child.name,
                #    child.postrank))
                #print("tm[:, {}].shape {}".format(
                #    child.postrank,
                #    tm[:, child.postrank].shape))

                #print("node.state.shape {}".format(
                #    node.state.shape))
                # [sample_size, n_dim, x_dim]

                partials= torch.einsum("bij,bcjk->bcik",
                        tm[:, child.postrank],
                        child.state.unsqueeze(-1)).squeeze(
                                -1).clamp(min=0., max=1.)
                
                #FIXME: Another alternative to compute partials
                # Does not work correctly when sample_size>1
                #partials = (tm[:, child.postrank] @
                #        child.state.unsqueeze(-1)).squeeze(
                #                -1).clamp(min=0., max=1.)
                #print("partials {}".format(partials.shape)) 
                # [sample_size, n_dim, x_dim]

                node.state *= partials
                #print("node {}\t{}".format(node.name,
                #    node.state.shape)) 
                # [sample_size, n_dim, x_dim]

            scaler = torch.sum(node.state, -1).unsqueeze(-1)
            #scaler = torch.max(node.state,
            #        -1).values.unsqueeze(-1)

            #print("scaler shape {}".format(scaler.shape))
            node.state /= scaler
            scaler_list.append(scaler)

            #print("\npi shape {}".format(pi.shape))
            # [sample_size, 1, x_dim]
            
            #  print(pi)        

            #  print("\nroot {}".format( arbre.state.shape))
            # [sample_size, n_dim, x_dim, 1]
            #  print(arbre.state)

            #b  i  j            b   c   j   k       
            #  pi unseqz(-2)   : [sample_size, 1, x_dim]  root unseqz(-1) : [sample_size, n_dim, x_dim, 1]

            #logl = torch.einsum("bij,bcjk->bcik", (pi.unsqueeze(-2),
            #    arbre.state.unsqueeze(-1))).log().mean(0).flatten()

    scaler_list.append(torch.einsum("bij,bcjk->bcik",
        (pi.unsqueeze(-2),
            arbre.state.unsqueeze(-1))).squeeze(-1))
    #scaler_list.append((pi @
    #    arbre.state.unsqueeze(-1)).squeeze(-1)) 

    #logl = torch.sum(torch.log(torch.stack(scaler_list)),
    #        dim=0).mean(0).flatten()
    logl = torch.sum(torch.log(torch.stack(scaler_list)),
            dim=0).squeeze(-1)
    #print("\nlogl")
    #print(logl.shape) #
    #print(logl)

    return logl

def pruning_known_ancestors(arbre, x, a, tm, pi):

        # Assign each node its sites
        for node in arbre.traverse("postorder"):
            if node.is_leaf():
                node.sites = x[:, :, node.postrank, :]
                # [sample_size, n_dim, x_dim]
            else:
                node.sites = a[:, :, node.ancestral_postrank,:]
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
                    #print("tm[:, {}].shape {}".format(child.postrank, tm[:, child.postrank].shape)) # [sample_size, x_dim, x_dim]
                    #print("child.sites.shape {}".format(child.sites.shape)) # [sample_size, n_dim, x_dim]
                    
                    partials = torch.einsum("bcij,bjk->bcik", (child.sites.unsqueeze(-2), tm[:, child.postrank])).squeeze(-2).clamp(min=0., max=1.)
                    #print("partials {}".format(partials.shape))  # [sample_size, n_dim, x_dim]
                    #print(partials)
                    #print()
                    #print("node.sites.shape {}".format(node.sites.shape)) # [sample_size, n_dim, x_dim]
                    #print(node.sites)
                     
                    #ll_cumul += (partials.squeeze().dot(node.sites.squeeze())).log()
                    ll_cumul += torch.log(torch.einsum("bij,bij->bi", (partials, node.sites))) #.log
                    #print("ll_cumul {}".format(ll_cumul.shape))  # [sample_size, n_dim, x_dim]

        logl = ll_cumul + log_pi_a
        #print("logl {}".format(logl.shape))

        return logl
