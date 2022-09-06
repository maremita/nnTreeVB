#################################
##                             ##
##         nnTreeVB            ##
##  2022 (C) Amine Remita      ##
##                             ##
#################################

# This code is adapted from evoABCmodels.py module of evoVGM
# https://github.com/maremita/evoVGM/blob/main/evoVGM/models/evoABCmodels.py

from nnTreeVB.utils import timeSince, dict_to_numpy
from nnTreeVB.utils import check_finite_grads
from nnTreeVB.utils import get_grad_stats
from nnTreeVB.utils import get_weight_stats
from nnTreeVB.utils import apply_on_submodules

from abc import ABC 
import time

import numpy as np
import torch
from scipy.spatial.distance import hamming

__author__ = "amine remita"

__all__ = ["BaseTreeVB"]


class BaseTreeVB(ABC):

    def sample(self,
            tree,
            sites,
            site_counts,
            elbo_type="elbo",
            latent_sample_size=1,
            sample_temp=0.1):
            #alpha_kl=0.001,

        with torch.no_grad():
            if site_counts == None:
                site_counts = torch.ones(sites.shape[0]).to(
                        self.device_)
            # Don't shuffle sites
            return self(
                    tree,
                    sites,
                    site_counts,
                    elbo_type=elbo_type,
                    latent_sample_size=latent_sample_size,
                    sample_temp=sample_temp, 
                    #alpha_kl=alpha_kl, 
                    shuffle_sites=False)

    def fit(self,
            tree,
            X_train,
            X_train_counts,
            elbo_type="elbo",
            latent_sample_size=1,
            sample_temp=0.1,
            #alpha_kl=0.001,
            max_iter=100,
            optim="adam",
            optim_learning_rate=0.005, 
            optim_weight_decay=0.1,
            X_val=None,
            # If not None, a validation stpe will be done
            X_val_counts=None,
            A_val=None, 
            # (np.ndarray) Embedded ancestral sequence which was used
            # to sample the estimates. If not None, it will be used only
            # to report its distance with the inferred ancestral
            # sequence during calidation
            # It is not used during the inference.
            save_fit_history=False,
            # Save estimates for each epoch of fitting step
            save_val_history=False,
            # Save estimates for each epoch of validation step
            #keep_fit_vars=False,
            # Save estimate variances from fitting step
            #keep_val_vars=False,
            # Save estimate variances from validation step
            save_grad_stats=False,
            # Save statistics of gradients
            save_weight_stats=False,
            # Save statistics of weights
            verbose=None):

        if optim == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), 
                    lr=optim_learning_rate,
                    weight_decay=optim_weight_decay)
        else:
            optimizer = torch.optim.SGD(self.parameters(),
                    lr=optim_learning_rate,
                    weight_decay=optim_weight_decay)

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

        if X_val_counts is not None: N_val_dim = X_val_counts.sum()
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

        optim_nb = 0
        for epoch in range(1, max_iter + 1):

            fit_time = time.time()
            optimizer.zero_grad()
            try:
                fit_dict = self(
                        tree,
                        X_train,
                        X_train_counts,
                        elbo_type=elbo_type,
                        latent_sample_size=latent_sample_size,
                        sample_temp=sample_temp,
                        #alpha_kl=alpha_kl,
                        shuffle_sites=True)

                elbo = fit_dict["elbo"]
                lls = fit_dict["logl"].cpu()
                lps = fit_dict["logprior"].cpu()
                lqs = fit_dict["logq"].cpu()
                kls = fit_dict["kl_qprior"].cpu()

                loss = - elbo
                loss.backward()

                f = check_finite_grads(self, epoch, verbose=False)

                # Optimize if gradients do not have nan values
                if f and torch.isfinite(elbo):
                    optimizer.step()
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
                                tree,
                                X_val,
                                X_val_counts, 
                                elbo_type=elbo_type,
                                latent_sample_size=latent_sample_size,
                                sample_temp=sample_temp)
                                #alpha_kl=alpha_kl,

                        val_dict = dict_to_numpy(val_dict)
                        elbo_val = val_dict["elbo"]
                        lls_val = val_dict["logl"]
                        lps_val = val_dict["logprior"]
                        lqs_val = val_dict["logq"]
                        kls_val = val_dict["kl_qprior"]

                    except Exception as e:
                        print("\nStopping training at epoch {}"\
                                " because of an exception in"\
                                " sample()\n".format(epoch))
                        print(e)
                        #TODO not break if there is an exception
                        break

                    ret["total_val_time"] += time.time() - val_time

                ## printing
                if verbose:
                    if epoch % 10 == 0:
                        chaine = "{}\t Train Epoch: {} \t"\
                                " ELBO: {:.3f}\t Lls {:.3f}\t KLs "\
                                "{:.3f}".format(timeSince(start),
                                        epoch, elbo.item(), 
                                        lls.item(), kls.item())

                        if X_val is not None:
                            chaine += "\nELBO_Val: {:.3f}\t"\
                                    " Lls_Val {:.3f}\t KLs "\
                                    "{:.3f}".format(
                                            elbo_val.item(),
                                            lls_val.item(), 
                                            lps_val.item(),
                                            lqs_val.item(),
                                            kls_val.item())
                        print(chaine, end="\r")

                ## Adding measure values to lists if all is OK
                ret["elbos_list"].append(elbo.item())
                ret["lls_list"].append(lls.item())
                ret["lps_list"].append(lps.item())
                ret["lqs_list"].append(lqs.item())
                ret["kls_list"].append(kls.item())

                if save_fit_history:
                    fit_estim = dict()
                    for estim in ["b", "t", "b1", "r", "f", "k"]:
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
                    ret["elbos_val_list"].append(elbo_val.item())
                    ret["lls_val_list"].append(lls_val.item())
                    ret["lps_val_list"].append(lps_val.item())
                    ret["lqs_val_list"].append(lqs_val.item())
                    ret["kls_val_list"].append(kls_val.item())

                    if save_val_history:
                        val_estim = dict()
                        for estim in ["b", "t", "b1", "r", "f", "k"]:
                            if estim in val_dict:
                                val_estim[estim]=val_dict[estim]

                        # Generated sequences and inferred ancestral
                        # sequences are not saved for each epoch
                        # because they consume a lot of space.
                        # Instead of that, we report their distances
                        # with actual sequences.
 
                        # Compute Hamming and average Euclidean 
                        # distances
                        # between X_val and generated sequences
                        if "x" in val_dict:
                            xrecons = val_dict["x"]
                            x_ham_dist = np.array([hamming(
                                xrecons[:,i,:].argmax(axis=1),
                                np_X_val[:,i,:].argmax(axis=1))\
                                        for i in range(
                                            np_X_val.shape[1])])
                            x_euc_dist = np.linalg.norm(
                                    xrecons -np_X_val,
                                    axis=2).mean(0)
                            val_estim['x_hamming'] = x_ham_dist
                            val_estim['x_euclidean'] = x_euc_dist
 
                        # Compute Hamming and average Euclidean 
                        # distances
                        # between actual ancestral sequence and
                        # inferred ancestral sequence
                        if A_val is not None and "a" in val_dict:
                            estim_ancestor = val_dict["a"]
                            a_ham_dist = np.array([hamming(
                                    estim_ancestor.argmax(axis=1),
                                    A_val.argmax(axis=1))])
                            a_euc_dist = np.array([np.linalg.norm(
                                    estim_ancestor - A_val, 
                                    axis=1).mean()])
                            val_estim['a_hamming'] = a_ham_dist
                            val_estim['a_euclidean'] = a_euc_dist

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
                ret["elbos_val_list"]=np.array(ret["elbos_val_list"])
                ret["lls_val_list"] = np.array(ret["lls_val_list"]) 
                ret["lps_val_list"] = np.array(ret["lps_val_list"])
                ret["lqs_val_list"] = np.array(ret["lqs_val_list"])
                ret["kls_val_list"] = np.array(ret["kls_val_list"])

        optim_rate = optim_nb/max_iter

        if verbose >= 2:
            print("\nOptimization rate {}".format(optim_rate))

        return ret
