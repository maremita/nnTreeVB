from nnTreeVB.models.vb_models import BaseTreeVB
from nnTreeVB.models.vb_models import VB_Encoder
from nnTreeVB.models.distributions import build_distribution
from nnTreeVB.models.evo_models import build_transition_matrix
from nnTreeVB.models.evo_models import pruning_rescaled
#from nnTreeVB.models.evo_models import pruning
from nnTreeVB.utils import sum_log_probs
from nnTreeVB.utils import sum_kls
from nnTreeVB.utils import check_sample_size

import math
import copy

import torch
import torch.nn as nn

__author__ = "Amine Remita"


class VB_nnTree(nn.Module, BaseTreeVB):
    def __init__(
            self,
            tree,
            #
            subs_model="gtr", # jc69 | k80 | hky | gtr
            # #################################################
            # Branch lengths encoder
            # Prior distribution:
            b_prior_dist="gamma",
            b_prior_params=[0.2, 0.2],
            b_learn_prior=False,
            # Variational distribution:
            # fixed | gamma | lognormal | normal
            b_var_dist="gamma", 
            b_var_params=[0.1, 0.1], 
            # if not nn: list of 2 floats
            # if nn: list of 2 floats, uniform, normal or False
            # if fixed: tensor
            #
            b_var_transform=None,
            # #################################################
            # Total tree length
            # Prior distribution:
            t_prior_dist="gamma",
            t_prior_params=[0.2, 0.2],
            t_learn_prior=False,
            # Variational distribution:
            # fixed | gamma | lognormal | normal
            t_var_dist="gamma", 
            t_var_params=[0.1, 0.1], 
            # if not nn: list of 2 floats
            # if nn: list of 2 floats, uniform, normal or False
            # if fixed: tensor
            #
            t_var_transform=None,
            # #################################################
            # GTR rates encoder args
            # Prior distribution:
            r_prior_dist="dirichlet",
            r_prior_params=[1.]*6,
            r_learn_prior=False,
            # Variational distribution:
            # fixed | dirichlet | normal
            r_var_dist="dirichlet_ind",
            r_var_params=[1.]*6, 
            # if not nn: list of 6 floats
            # if nn: list of 6 floats, uniform, normal or False
            # if fixed: tensor
            #
            r_var_transform=None,
            # #################################################
            # GTR frequencies encoder args
            # Prior distribution:
            f_prior_dist="dirichlet",
            f_prior_params=[1.]*4,
            f_learn_prior=False,
            # Variational distribution:
            # fixed | dirichlet | normal
            f_var_dist="dirichlet_ind",  # 
            f_var_params=[1.]*4, 
            # if not nn: list of 6 floats
            # if nn: list of 6 floats, uniform, normal or False
            # if fixed: tensor
            #
            f_var_transform=None,
            # #################################################
            # Kappa encoder args
            # Prior distribution:
            k_prior_dist="gamma",
            k_prior_params=[0.1, 0.1],
            k_learn_prior=False,
            # Variational distribution:
            # fixed | gamma | lognormal | normal
            k_var_dist="gamma_ind",
            k_var_params=[0.1, 0.1], 
            # if not nn: list of 2 floats
            # if nn: list of 2 floats, uniform, normal or False
            # if fixed: tensor
            #
            k_var_transform=None,
            # #################################################
            # Following parameters are needed if nn
            h_dim=16,
            nb_layers=3,
            bias_layers=True,     # True or False
            activ_layers="relu",  # relu, tanh, or False
            dropout_layers=0.,
            device=torch.device("cpu")):
 
        super().__init__()

        self.tree = copy.deepcopy(tree)

        self.m_dim = len(self.tree.get_leaf_names())
        self.b_dim = len(self.tree.get_edges()) - 1

        self.t_dim = 1
        self.r_dim = 6
        self.f_dim = 4
        self.k_dim = 1

        self.subs_model = subs_model
        self.device_ = device

        common_args = dict(
                h_dim=h_dim,
                nb_layers=nb_layers,
                bias_layers=bias_layers,
                activ_layers=activ_layers,
                dropout_layers=dropout_layers,
                device=self.device_)

        self.b_compound = False
        if "dirichlet" in b_var_dist:
            self.b_compound = True

        # Initialize branch length prior distribution
        self.b_dist_p = build_distribution(
                [self.b_dim],
                [self.b_dim],
                dist_type=b_prior_dist,
                init_params=b_prior_params,
                learn_params=b_learn_prior,
                transform_dist=None,
                **common_args)

        # Initialize branch length variational distribution
        self.b_dist_q = build_distribution(
                [self.b_dim],
                [self.b_dim],
                dist_type=b_var_dist,
                init_params=b_var_params,
                learn_params=True,
                transform_dist=b_var_transform,
                **common_args)
 
        # Initialize branch length encoder
        self.b_encoder = VB_Encoder(
                self.b_dist_p, self.b_dist_q)

        if self.b_compound:
            # Using a Compound Dirichlet Gamma distribution
            # Initialize tree length prior distribution
            self.t_dist_p = build_distribution(
                    [self.t_dim],
                    [self.t_dim],
                    dist_type=t_prior_dist,
                    init_params=t_prior_params,
                    learn_params=t_learn_prior,
                    transform_dist=None,
                    **common_args)

            # Initialize tree length variational distribution
            self.t_dist_q = build_distribution(
                    [self.t_dim],
                    [self.t_dim],
                    dist_type=t_var_dist,
                    init_params=t_var_params,
                    learn_params=True,
                    transform_dist=t_var_transform,
                    **common_args)
 
            # Initialize tree length encoder
            self.t_encoder = VB_Encoder(
                    self.t_dist_p, self.t_dist_q)

        if self.subs_model in ["gtr"]:
            # Initialize rates prior distribution
            self.r_dist_p = build_distribution(
                    [self.r_dim],
                    [self.r_dim],
                    dist_type=r_prior_dist,
                    init_params=r_prior_params,
                    learn_params=r_learn_prior,
                    transform_dist=None,
                    **common_args)

            # Initialize rates variational distribution
            self.r_dist_q = build_distribution(
                    [self.r_dim],
                    [self.r_dim],
                    dist_type=r_var_dist,
                    init_params=r_var_params,
                    learn_params=True,
                    transform_dist=r_var_transform,
                    **common_args)

            # Initialize rates encoder
            self.r_encoder = VB_Encoder(
                    self.r_dist_p, self.r_dist_q)

        if self.subs_model in ["hky", "gtr"]:
            # Initialize frequencies prior distribution
            self.f_dist_p = build_distribution(
                    [self.f_dim],
                    [self.f_dim],
                    dist_type=f_prior_dist,
                    init_params=f_prior_params,
                    learn_params=f_learn_prior,
                    transform_dist=None,
                    **common_args)

            # Initialize frequencies variational distribution
            self.f_dist_q = build_distribution(
                    [self.f_dim],
                    [self.f_dim],
                    dist_type=f_var_dist,
                    init_params=f_var_params,
                    learn_params=True,
                    transform_dist=f_var_transform,
                    **common_args)

            # Initialize frequencies encoder
            self.f_encoder = VB_Encoder(
                    self.f_dist_p, self.f_dist_q)

        if self.subs_model in ["k80", "hky"]:
            # Initialize kappa prior distribution
            self.k_dist_p = build_distribution(
                    [self.k_dim],
                    [self.k_dim],
                    dist_type=k_prior_dist,
                    init_params=k_prior_params,
                    learn_params=k_learn_prior,
                    transform_dist=None,
                    **common_args)

            # Initialize kappa variational distribution
            self.k_dist_q = build_distribution(
                    [self.k_dim],
                    [self.k_dim],
                    dist_type=k_var_dist,
                    init_params=k_var_params,
                    learn_params=True,
                    transform_dist=k_var_transform,
                    **common_args)
            
            # Initialize kappa encoder
            self.k_encoder = VB_Encoder(
                    self.k_dist_p, self.k_dist_q)

    def forward(self,
            sites, 
            site_counts,
            elbo_type="elbo",
            sample_size=torch.Size([1]),
            alpha_kl=1.,
            shuffle_sites=True):

        # returned dict
        ret_values = dict()

        #
        sample_size = check_sample_size(sample_size)

        #
        elbos = ["elbo", "elbo_iws", "elbo_kl"]
        elbo_type = elbo_type.lower()
        if elbo_type not in elbos:
            print("Warning {} is not a valid type for"\
                    " elbo".format(elbo_type))
            print("elbo_type is set to elbo")
            elbo_type = "elbo"

        if len(sample_size) == 1 and elbo_type=="elbo_iws":
            elbo_type = "elbo"
        elif len(sample_size) == 2 and elbo_type!="elbo_iws":
            elbo_type = "elbo_iws"

        elbo_kl = elbo_type=="elbo_kl"
        elbo_iws = elbo_type=="elbo_iws"

        #
        pi = (torch.ones(4)/4).expand([*list(sample_size), 4])
        tm_args = dict()

        # Sum joint log probs by samples [True] or 
        # Sum averages of each log probs by samples [False]
        # Both methods should give the same result
        sum_by_samples = True

        logpriors = []
        logqs = []
        kl_qpriors = []

        n_dim, m_dim, x_dim = sites.shape

        assert self.m_dim == m_dim

        ## Inference and sampling
        ## ######################

        # Sample b from q_d and compute log prior, log q
        b_logprior, b_logq, b_kl, b_samples = self.b_encoder(
                sample_size=sample_size,
                KL_gradient=elbo_kl)

        #print("b_logprior {}".format(b_logprior.shape))
        #print("b_logq {}".format(b_logq.shape))
        #print("b_kl {}".format(b_kl.shape))

        logpriors.append(b_logprior)
        logqs.append(b_logq)
        kl_qpriors.append(b_kl)

        ret_values["b"] = b_samples.detach().numpy()
        tm_args["b"] = b_samples
        #print("b_samples.shape {}".format(b_samples.shape))

        if self.b_compound:
            # Sample t from q_d and compute log prior, log q
            t_logprior, t_logq, t_kl, t_samples =\
                    self.t_encoder(
                            sample_size=sample_size,
                            KL_gradient=elbo_kl)

            bt_samples = (b_samples * t_samples)
            #print("t_samples.shape {}".format(
            #    t_samples.shape))
            #print("bt_samples.shape {}".format(
            #    bt_samples.shape))
            #print("t_logprior {}".format(t_logprior.shape))
            #print("t_logq {}".format(t_logq.shape))
            #print("t_kl {}".format(t_kl.shape))

            logpriors.append(t_logprior)
            logqs.append(t_logq)
            kl_qpriors.append(t_kl)

            ret_values["t"] = t_samples.detach().numpy()
            ret_values["b1"] = b_samples.detach().numpy()
            ret_values["b"] = bt_samples.detach().numpy()
            tm_args["b"] = bt_samples

        if self.subs_model in ["gtr"]:
            # Sample r from q_r and compute log prior, log q
            r_logprior, r_logq, r_kl, r_samples =\
                    self.r_encoder(
                            sample_size=sample_size,
                            KL_gradient=elbo_kl)

            #print("r_samples {}".format(r_samples.shape))
            #print("r_logprior {}".format(r_logprior.shape))
            #print("r_logq {}".format(r_logq.shape))
            #print("r_kl {}".format(r_kl.shape))

            logpriors.append(r_logprior)
            logqs.append(r_logq)
            kl_qpriors.append(r_kl)

            ret_values["r"] = r_samples.detach().numpy()
            tm_args["rates"] = r_samples

        if self.subs_model in ["hky", "gtr"]:
            # Sample f from q_f and compute log prior, log q
            f_logprior, f_logq, f_kl, f_samples =\
                    self.f_encoder(
                            sample_size=sample_size,
                            KL_gradient=elbo_kl)

            #print("f_logprior {}".format(f_logprior.shape))
            #print("f_logq {}".format(f_logq.shape))
            #print("f_kl {}".format(f_kl.shape))

            logpriors.append(f_logprior)
            logqs.append(f_logq)
            kl_qpriors.append(f_kl)

            ret_values["f"] = f_samples.detach().numpy()
            pi = f_samples
            tm_args["freqs"] = f_samples

        if self.subs_model in ["k80", "hky"]:
            # Sample k from q_d and compute log prior, log q
            k_logprior, k_logq, k_kl, k_samples =\
                    self.k_encoder(
                            sample_size=sample_size,
                            KL_gradient=elbo_kl)

            #print("k_logprior {}".format(k_logprior.shape))
            #print("k_logq {}".format(k_logq.shape))
            #print("k_kl {}".format(k_kl.shape))

            logpriors.append(k_logprior)
            logqs.append(k_logq)
            kl_qpriors.append(k_kl)

            ret_values["k"] = k_samples.detach().numpy()
            tm_args["kappa"] = k_samples

        # Compute joint logprior, logq and kl
        logprior = sum_log_probs(logpriors, sample_size,
                sum_by_samples)
        logq = sum_log_probs(logqs, sample_size,
                sum_by_samples)
        kl_qprior = sum_kls(kl_qpriors)

        #print("logprior {}".format(logprior.shape))
        #print("logq {}".format(logq.shape))
        #print("kl_qprior {}".format(kl_qprior.shape))

        ## Compute logl
        ## ############
        tm = build_transition_matrix(self.subs_model, tm_args)
        sites_expanded = sites.expand(
                [*list(sample_size), n_dim, m_dim, x_dim])

        #logl = pruning(self.tree,
        #        sites_expanded, tm, pi) * site_counts

        logl = pruning_rescaled(self.tree,
                sites_expanded, tm, pi) * site_counts

        if sum_by_samples:
            logl = (logl).sum(-1)
        else:
            logl = (logl).mean(0).sum(0)

        # Compute the Elbo
        if elbo_kl:
            elbo = torch.mean(logl, 0) - (alpha_kl * kl_qprior)
        else:
            elbo = logl + logprior - logq

            if elbo_iws:
                nb_sample = logl.shape[-1]
                elbo = torch.logsumexp(elbo, dim=-1,
                        keepdim=True) - math.log(nb_sample)

            elbo = elbo.mean()

        # returned dict
        ret_values["elbo"]=elbo
        ret_values["logl"]=logl.mean()
        ret_values["logprior"]=logprior.mean()
        ret_values["logq"]=logq.mean()
        ret_values["kl_qprior"]=kl_qprior

        return ret_values
