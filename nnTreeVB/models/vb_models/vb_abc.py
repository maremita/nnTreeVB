from nnTreeVB.utils import timeSince, dict_to_numpy
from nnTreeVB.utils import get_grad_stats
from nnTreeVB.utils import get_weight_stats
from nnTreeVB.utils import apply_on_submodules
from nnTreeVB.checks import check_finite_grads

from abc import ABC 
import time

import numpy as np
import torch

__author__ = "amine remita"

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
                        self.device_)
            return self(
                    X,
                    X_counts,
                    elbo_type=elbo_type,
                    sample_size=nb_samples,
                    alpha_kl=alpha_kl) 

    def fit(self,
            X:torch.Tensor,
            X_counts:torch.Tensor,
            elbo_type: str = "elbo",
            grad_samples=torch.Size([1]),
            nb_samples=torch.Size([1]),
            alpha_kl=1.,
            max_iter:int = 100,
            optim="adam",
            optim_learning_rate=0.005, 
            optim_weight_decay=0.1,
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
            verbose=None):
 
        # Optimizer configuration
        optim_algo = torch.optim.SGD
        if optim == 'adam':
            optim_algo = torch.optim.Adam

        optim_params = self.parameters()
        lr_default = 0.01

        if isinstance(optim_learning_rate, dict):
            optim_params = []

            for name, encoder in self.named_children():
                if name in optim_learning_rate:
                    ps = {'params': encoder.parameters(),
                            'lr': optim_learning_rate[name]}
                    optim_params.append(ps)

            if 'default' in optim_learning_rate:
                lr_default = optim_learning_rate['default']

        elif isinstance(optim_learning_rate, (int, float)):
            lr_default = optim_learning_rate

        optimizer = optim_algo(
                optim_params, 
                lr=lr_default,
                weight_decay=optim_weight_decay)

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

        m_axis = 0
        if len(nb_samples) == 2:
            m_axis = (0,1)

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
                                nb_samples=nb_samples)
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
                if verbose:
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
                        if verbose >= 2:
                            chaine += "\n"
                            for estim in ["b", "t", "b1",\
                                    "r", "f", "k"]:
                                if estim in fit_dict:
                                    estim_vals = fit_dict[
                                            estim].mean(m_axis).squeeze() 
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
                    for estim in ["b", "t", "b1", "r",\
                            "f", "k"]:
                        if estim in fit_dict:
                            fit_estim[estim] = fit_dict[estim]
                    ret["fit_estimates"].append(fit_estim)

                if save_grad_stats:
                    ret["grad_stats"].append(
                            apply_on_submodules(
                                get_grad_stats, self))

                if save_weight_stats:
                    ret["weight_stats"].append(
                            apply_on_submodules(
                                get_weight_stats, self))

                if X_val is not None:
                    ret["elbos_val_list"].append(
                            elbo_val.item())
                    ret["lls_val_list"].append(lls_val.item())
                    ret["lps_val_list"].append(lps_val.item())
                    ret["lqs_val_list"].append(lqs_val.item())
                    ret["kls_val_list"].append(kls_val.item())

                    if save_val_history:
                        val_estim = dict()
                        for estim in ["b", "t", "b1", "r",\
                                "f", "k"]:
                            if estim in val_dict:
                                val_estim[estim]=val_dict[
                                        estim]

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
