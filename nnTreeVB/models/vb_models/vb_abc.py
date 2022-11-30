from nnTreeVB.utils import timeSince, dict_to_numpy
from nnTreeVB.utils import get_named_grad_stats
from nnTreeVB.utils import get_named_weight_stats
from nnTreeVB.utils import apply_on_submodules
from nnTreeVB.utils import compute_estim_stats
from nnTreeVB.checks import check_finite_grads
from nnTreeVB.checks import check_sample_size

from abc import ABC 
import time

import numpy as np
import torch

__author__ = "amine remita"

estim_names = ["b", "t", "b1", "r", "f", "k"]

# This code is adapted from evoABCmodels.py module of evoVGM
# https://github.com/maremita/evoVGM/blob/main/evoVGM/models/evoABCmodels.py
class BaseTreeVB(ABC):

    def sample(self,
            X:torch.Tensor,
            X_counts:torch.Tensor,
            elbo_type="elbo",
            nb_samples=torch.Size([1]),
            alpha_kl=1.):

        with torch.no_grad():
            if X_counts == None:
                X_counts = torch.ones(X.shape[0]).to(
                        device=self.device_,
                        dtype=self.dtype_)
            ret_dict = self(
                    X,
                    X_counts,
                    elbo_type=elbo_type,
                    sample_size=nb_samples,
                    alpha_kl=alpha_kl)

            for estim in ret_dict:
                if not isinstance(ret_dict[estim],
                        np.ndarray):
                    ret_dict[estim] = ret_dict[estim].cpu(
                            ).numpy()
            return ret_dict

    def fit(self,
            X:torch.Tensor,
            X_counts:torch.Tensor,
            elbo_type: str = "elbo",
            grad_samples=torch.Size([1]),
            val_samples=torch.Size([1]),
            alpha_kl=1.,
            max_iter:int = 100,
            optimizer="adam",
            learning_rate=0.1, 
            weight_decay=0.,
            scheduler_lambda=lambda e:1.,
            X_val=None,
            # If not None, a validation stpe will be done
            X_val_counts=None,
            save_fit_history=False,
            # Save estimates for each epoch of fitting step
            save_val_history=False,
            #Save estimates for each epoch of validation step
            save_grad_stats=False,
            # Save statistics of gradients
            save_weight_stats=False,
            # Save statistics of weights
            verbose=None,
            **kwargs):

        # Optimizer configuration
        optim_algo = torch.optim.SGD
        if optimizer == 'adam':
            optim_algo = torch.optim.Adam

        lr_default = 0.1

        if isinstance(learning_rate, dict):
            if 'default' in learning_rate:
                lr_default = learning_rate['default']

            optim_params = []

            for name, params in self.named_parameters():
                main_name = name.split(".")[0]
                if main_name in learning_rate:
                    _lr = learning_rate[main_name]
                else:
                    _lr = lr_default 

                optim_params.append(
                        {'params': params, 'lr': _lr,
                            'name': name})

            # TODO I think this is useless now:
            if len(optim_params) == 0:
                optim_params = list(self.parameters())

        elif isinstance(learning_rate, (int, float)):
            optim_params = list(self.parameters())
            lr_default = float(learning_rate)

        else:
            raise ValueError("Learning rate must be of"\
                    " type dict, int or float")

        optimizer = optim_algo(
                optim_params, 
                lr=lr_default,
                weight_decay=weight_decay)

        # Scheduler configuration
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=scheduler_lambda)

        # dict to be returned
        # it will contain the results of fit/val
        ret = {}

        # Time for printing
        start = time.time()

        # Times of fitting and validation steps
        # without pre/post processing tasks
        ret["total_fit_time"] = 0
        ret["total_val_time"] = 0
        ret["optimized"] = []

        if X_val_counts is not None:
            N_val_dim = X_val_counts.sum()
        elif X_val is not None: N_val_dim = X_val.shape[0]
 
        ret["elbos_list"] = []
        ret["lls_list"] = []
        ret["lps_list"] = []
        ret["lqs_list"] = []
        ret["kls_list"] = []

        if save_fit_history: ret["fit_estimates"] = []
        if save_val_history: ret["val_estimates"] = []
        if save_grad_stats: ret["grad_stats"] = []
        if save_weight_stats: ret["weight_stats"] = []

        if X_val is not None:
            np_X_val = X_val.cpu().numpy()
            ret["elbos_val_list"] = []
            ret["lls_val_list"] = []
            ret["lps_val_list"] = []
            ret["lqs_val_list"] = []
            ret["kls_val_list"] = []

        grad_samples = check_sample_size(grad_samples)
        val_samples = check_sample_size(val_samples)

        grad_axes = 0
        tgs = grad_samples[0] # total number of grad samples
        tvs = val_samples[0]  # total number of val samples

        if len(grad_samples) == 2:
            grad_axes = (0,1)
            tgs *= grad_samples[1]
        if len(val_samples) == 2:
            tvs *= val_samples[1]

        optim_nb = 0
        for epoch in range(1, max_iter + 1):

            fit_time = time.time()
            optimizer.zero_grad()
            try:
                fit_dict = self(
                        X,
                        X_counts,
                        elbo_type=elbo_type,
                        sample_size=grad_samples,
                        alpha_kl=alpha_kl)

                elbo = fit_dict["elbo"]
                lls = fit_dict["logl"].cpu()
                lps = fit_dict["logprior"].cpu()
                lqs = fit_dict["logq"].cpu()
                kls = fit_dict["kl_qprior"].cpu()

                loss = - elbo
                loss.backward()

                f = check_finite_grads(self, epoch,
                        verbose=False)

                # Optimize if gradients don't have nan values
                if f and torch.isfinite(elbo):
                    optimizer.step()
                    scheduler.step()
                    ret["optimized"].append(1.)
                    optim_nb += 1

                else:
                    ret["optimized"].append(0.)

            except Exception as e:
                print("\nStopping training at epoch {}"\
                        " because of an exception in"\
                        " fit()\n".format(epoch))
                print(e)
                #TODO not break if there is an exception
                break

            ret["total_fit_time"] += time.time() - fit_time

            with torch.no_grad():
                ## Validation
                if X_val is not None:
                    val_time = time.time()
                    try:
                        val_dict = self.sample(
                                X_val,
                                X_val_counts, 
                                elbo_type=elbo_type,
                                nb_samples=val_samples)
                                #alpha_kl=alpha_kl,

                        val_dict = dict_to_numpy(val_dict)
                        elbo_val = val_dict["elbo"]
                        lls_val = val_dict["logl"]
                        lps_val = val_dict["logprior"]
                        lqs_val = val_dict["logq"]
                        kls_val = val_dict["kl_qprior"]

                    except Exception as e:
                        print("\nStopping training at epoch ""\
                                {}"\
                                " because of an exception in"\
                                " sample()\n".format(epoch))
                        print(e)
                        #TODO not break if there is exception
                        break

                    ret["total_val_time"] +=\
                            time.time() - val_time

                ## printing
                if verbose >= 2:
                    if epoch % 10 == 0:
                        chaine = "{}\tEpoch: {}"\
                                "\tELBO: {:.3f}"\
                                "\tLogL: {:.3f}"\
                                "\tLogP: {:.3f}"\
                                "\tLogQ: {:.3f}"\
                                "\tKL: {:.3f}".format(
                                        timeSince(start),
                                        epoch, 
                                        elbo.item(), 
                                        lls.item(),
                                        lps.item(),
                                        lqs.item(),
                                        kls.item())

                        if X_val is not None:
                            chaine += "\nVal\tELBO: {:.3f}"\
                                    "\tLogL: {:.3f}"\
                                    "\tLogP: {:.3f}"\
                                    "\tLogQ: {:.3f}"\
                                    "\tKL: {:.3f}".format(
                                            elbo_val.item(),
                                            lls_val.item(), 
                                            lps_val.item(),
                                            lqs_val.item(),
                                            kls_val.item())
                        if verbose >= 3:
                            chaine += "\n"
                            for estim in estim_names:
                                if estim in fit_dict:
                                    estim_vals = fit_dict[
                                        estim].mean(grad_axes
                                                ).squeeze()
                                    chaine += estim + ": "\
                                            +str(estim_vals)
                                    if estim == "b" and "t"\
                                            not in fit_dict:
                                        chaine += "\nt: "+ str(
                                            estim_vals.sum()) 
                                    chaine += "\n"
                        print(chaine, end="\n")

                ## Adding measure values to lists if all is OK
                ret["elbos_list"].append(elbo.item())
                ret["lls_list"].append(lls.item())
                ret["lps_list"].append(lps.item())
                ret["lqs_list"].append(lqs.item())
                ret["kls_list"].append(kls.item())

                if save_fit_history:
                    fit_estim = dict()
                    for estim in estim_names:
                        if estim in fit_dict:
                            estim_stats = compute_estim_stats(
                                    fit_dict[estim].reshape(
                                        tgs, -1),
                                    confidence=0.95,
                                    axis=0)
                            fit_estim[estim] = estim_stats
                    ret["fit_estimates"].append(fit_estim)

                if save_grad_stats:
                    ret["grad_stats"].append(
                            apply_on_submodules(
                                get_named_grad_stats, self))

                if save_weight_stats:
                    ret["weight_stats"].append(
                            apply_on_submodules(
                                get_named_weight_stats, self))

                if X_val is not None:
                    ret["elbos_val_list"].append(
                            elbo_val.item())
                    ret["lls_val_list"].append(lls_val.item())
                    ret["lps_val_list"].append(lps_val.item())
                    ret["lqs_val_list"].append(lqs_val.item())
                    ret["kls_val_list"].append(kls_val.item())

                    if save_val_history:
                        val_estim = dict()
                        for estim in estim_names:
                            if estim in val_dict:
                                estim_stats =\
                                    compute_estim_stats(
                                        val_dict[estim]\
                                                .reshape(
                                                    tvs, -1),
                                            confidence=0.95,
                                            axis=0)
                                val_estim[estim] = estim_stats
                        ret["val_estimates"].append(val_estim)
        # End of fitting/validating

        with torch.no_grad():
            # Convert to ndarray to facilitate post-processing
            ret["elbos_list"] = np.array(ret["elbos_list"])
            ret["lls_list"] = np.array(ret["lls_list"]) 
            ret["lps_list"] = np.array(ret["lps_list"])
            ret["lqs_list"] = np.array(ret["lqs_list"])
            ret["kls_list"] = np.array(ret["kls_list"])
            if X_val is not None:
                ret["elbos_val_list"]=np.array(
                        ret["elbos_val_list"])
                ret["lls_val_list"] = np.array(
                        ret["lls_val_list"]) 
                ret["lps_val_list"] = np.array(
                        ret["lps_val_list"])
                ret["lqs_val_list"] = np.array(
                        ret["lqs_val_list"])
                ret["kls_val_list"] = np.array(
                        ret["kls_val_list"])

        optim_rate = optim_nb/max_iter

        if verbose >= 2:
            print("\nOptimization rate {}".format(optim_rate))

        return ret
