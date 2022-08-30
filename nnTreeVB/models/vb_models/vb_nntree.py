from nnTreeVB.models.vb_models import BaseTreeVB
from nnTreeVB.models.vb_encoders import build_vb_encoder
from nnTreeVB.models.evo_models import build_transition_matrix
from nnTreeVB.models.evo_models import pruning_known_ancestors
from nnTreeVB.models.evo_models import pruning

import torch
import torch.nn as nn

__author__ = "Amine Remita"


class VB_nnTree(nn.Module, BaseTreeVB):
    def __init__(
            self,
            subs_model="gtr" # jc69 | k80 | hky | gtr
            # #####################################################
            # ancestor encoder
            predict_ancestors = False
            # categorical
            a_encoder_type="categorical",
            a_init_distr=[1., 1., 1., 1.], 
            # if not nn: list of 4 floats
            # if nn: list of 4 floats, uniform, normal or False
            a_hp=[1., 1., 1., 1.],
            # #####################################################
            # branch lengths
            # fixed | gamma | explogn | lognormal | dirichlet
            b_encoder_type="gamma", 
            b_init_distr=[0.1, 0.1], 
            # if not nn: list of 2 floats
            # if nn: list of 2 floats, uniform, normal or False
            b_hp=[0.2, 0.2],
            # #####################################################
            # Total tree length
            # fixed | gamma | explogn | lognormal
            t_encoder_type="gamma", 
            t_init_distr=[0.1, 0.1],
            # if not nn: list of 2 floats
            # if nn: list of 2 floats, uniform, normal or False
            t_hp=[0.2, 0.2],
            # #####################################################
            # gtr rates encoder args
            # fixed | dirichlet | nndirichlet
            r_encoder_type="dirichlet",
            r_init_distr=[1., 1., 1., 1., 1., 1.], 
            # if not nn: list of 6 floats
            # if nn: list of 6 floats, uniform, normal or False
            r_hp=[1., 1., 1., 1., 1., 1.],
            # #####################################################
            # gtr frequencies encoder args
            # fixed | dirichlet | nndirichlet
            f_encoder_type="dirichlet",  # 
            f_init_distr=[1., 1., 1., 1.], 
            # if not nn: list of 6 floats
            # if nn: list of 6 floats, uniform, normal or False
            f_hp=[1., 1., 1., 1.],
            # #####################################################
            # k encoder args
            k_encoder_type="gamma",  # gamma | explogn | lognormal
            k_init_distr=[0.1, 0.1], 
            # if not deep: list of 2 floats
            # if deep: list of 2 floats, uniform, normal or False
            k_hp=[0.1, 0.1],
            # #####################################################
            # Following parameters are needed id deep_encoder is True
            h_dim=16,
            nb_layers=3,
            bias_layers=True,     # True or False
            activ_layers="relu"):  # relu, tanh, or False

        self.predict_ancestors = predict_ancestors
        self.subs_model = subs_model

        if self.predict_ancestors: 
            # Initialize ancestral state encoder
            self.a_encoder = build_vb_encoder(
                    encoder_type=a_encoder_type,
                    init_distr=a_init_distr,
                    prior_hp=a_hp,
                    h_dim=h_dim,
                    nb_layers=nb_layers,
                    bias_layers=bias_layers,
                    activ_layers=activ_layers)

        # Initialize branch length encoder
        self.b_encoder = build_vb_encoder(
                encoder_type=b_encoder_type,
                init_distr=b_init_distr,
                prior_hp=b_hp,
                h_dim=h_dim,
                nb_layers=nb_layers,
                bias_layers=bias_layers,
                activ_layers=activ_layers)

        if b_encoder_type in ["dirichlet", "nndirichlet"]: 
            # Initialize tree length encoder
            # Using a Compound Dirichlet Gamma distribution
            self.t_encoder = build_vb_encoder(
                    encoder_type=t_encoder_type,
                    init_distr=t_init_distr,
                    prior_hp=t_hp,
                    h_dim=h_dim,
                    nb_layers=nb_layers,
                    bias_layers=bias_layers,
                    activ_layers=activ_layers)

        if self.subs_model in ["gtr"]:
            # Initialize rates encoder
            self.r_encoder = build_vb_encoder(
                    encoder_type=r_encoder_type,
                    init_distr=r_init_distr,
                    prior_hp=r_hp,
                    h_dim=h_dim,
                    nb_layers=nb_layers,
                    bias_layers=bias_layers,
                    activ_layers=activ_layers)

        if self.subs_model in ["hky", "gtr"]:
            # Initialize frequencies encoder
            self.f_encoder = build_vb_encoder(
                    encoder_type=f_encoder_type,
                    init_distr=f_init_distr,
                    prior_hp=f_hp,
                    h_dim=h_dim,
                    nb_layers=nb_layers,
                    bias_layers=bias_layers,
                    activ_layers=activ_layers)

        if self.subs_model in ["k80", "hky"]:
            # Initialize kappa encoder
            self.k_encoder = build_vb_encoder(
                    encoder_type=k_encoder_type,
                    init_distr=k_init_distr,
                    prior_hp=f_hp,
                    h_dim=h_dim,
                    nb_layers=nb_layers,
                    bias_layers=bias_layers,
                    activ_layers=activ_layers)

    def forward(self,
            tree,
            sites, 
            site_counts,
            elbo_type="elbo",
            latent_sample_size=10,
            sample_temp=0.1,
            #alpha_kl=0.001,
            shuffle_sites=True):

        # returned dict
        ret_values = dict()

        elbos = ["elbo", "elbo_iws", "elbo_kl"]
        elbo_type = elbo_type.lower()
        if elbo_type not in elbos:
            print("Warning {} is not a valid type for elbo".format(
                elbo_type))
            print("elbo_type is set to elbo")
            elbo_type = "elbo"

        elbo_kl = elbo_type=="elbo_kl"
        elbo_iws = elbo_type=="elbo_iws"

        tm_args = dict()

        logprior = torch.zeros(1).to(self.device_).detach()
        logq = torch.zeros(1).to(self.device_).detach()
        kl_qprior = torch.zeros(1).to(self.device_).detach()

        N = site_counts.sum().detach()

        ## Inference and sampling
        ## ######################
        if self.predict_ancestors: 
            # Sample a from q_d and compute log prior, log q
            a_logprior, a_logq, a_kl, a_samples = self.a_encoder(
                    sites,
                    sample_size=sample_size,
                    sample_temp=sample_temp,
                    KL_gradient=elbo_kl)

            logprior += a_logprior
            logq += a_logq
            kl_qprior += a_kl

        ret_values["a"] = a_samples.detach().numpy()

        # Sample b from q_d and compute log prior, log q
        b_logprior, b_logq, b_kl, b_samples = self.b_encoder(
                sample_size=sample_size,
                KL_gradient=elbo_kl)

        logprior += b_logprior
        logq += b_logq
        kl_qprior += b_kl
 
        ret_values["b"] = b_samples.detach().numpy()

        if b_encoder_type in ["dirichlet", "nndirichlet"]: 
            # Sample t from q_d and compute log prior, log q
            t_logprior, t_logq, t_kl, t_samples = self.t_encoder(
                    sample_size=sample_size,
                    KL_gradient=elbo_kl)

            bt_samples = b_samples * t_samples # update view ?

            logprior += t_logprior
            logq += t_logq
            kl_qprior += t_kl

            tm_args{"b"} = bt_samples
            ret_values["t"] = t_samples.detach().numpy()
            ret_values["bt"] = bt_samples.detach().numpy()

        else:
            tm_args{"b"} = b_samples

        if self.subs_model in ["gtr"]:
            # Sample r from q_r and compute log prior, log q
            r_logprior, r_logq, r_kl, r_samples = self.r_encoder(
                    sample_size=sample_size,
                    KL_gradient=elbo_kl)

            logprior += r_logprior
            logq += r_logq
            kl_qprior += r_kl
 
            tm_args{"rates"} = r_samples
            ret_values["r"] = r_samples.detach().numpy()

        if self.subs_model in ["hky", "gtr"]:
            # Sample f from q_f and compute log prior, log q
            f_logprior, f_logq, f_kl, f_samples = self.f_encoder(
                    sample_size=sample_size,
                    KL_gradient=elbo_kl)

            logprior += f_logprior
            logq += f_logq
            kl_qprior += f_kl

            tm_args{"freqs"} = f_samples
            ret_values["f"] = f_samples.detach().numpy()

        if self.subs_model in ["k80", "hky"]:
            # Sample k from q_d and compute log prior, log q
            k_logprior, k_logq, k_kl, k_samples = self.k_encoder(
                    sample_size=sample_size,
                    KL_gradient=elbo_kl)

            logprior += k_logprior
            logq += k_logq
            kl_qprior += k_kl
 
            tm_args{"kappa"} = k_samples
            ret_values["k"] = k_samples.detach().numpy()

        ## Compute logl
        ## ############
        tm = build_transition_matrix(self.subs_model, tm_args)

        if self.predict_ancestors:
            logl = pruning_known_ancestors(tree, sites, a_samples,
                    tm, f_samples)
            logl = (logl * site_counts).mean(0).sum(0, keepdim=True)

        else:
            logl = pruning(tree, sites, tm, f_samples)
            logl = (logl * site_counts).sum(0, keepdim=True)

        finit_inds = torch.isfinite(logl)

        # Compute the Elbo
        if elbo_kl:
            elbo = torch.mean(logl[finit_inds], 0) - kl
        else:
            elbo = logl[finit_inds] + logprior[finit_inds]\
                    - logq[finit_inds]

            if elbo_iws:
                nb_sample = logl[finit_inds].shape[0]
                elbo = torch.logsumexp(elbo, dim=0, keepdim=True) -\
                        math.log(nb_sample)
            else:
                elbo = elbo.mean(0)

        # returned dict
        ret_values["elbo"]=elbo,
        ret_values["logl"]=logl,
        ret_values["logprior"]=logprior,
        ret_values["logq"]=logq,
        ret_values["kl_qprior"]=kl,

        return ret_values
