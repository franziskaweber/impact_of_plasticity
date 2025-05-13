"""
This file contains various helper functions for the notebook 'impact_of_plasticity.ipynb'.
"""

import os
import json
import numpy as np
import scipy
import matplotlib.pyplot as plt
import reservoirpy as rpy
from reservoirpy.observables import rmse
from reservoirpy.nodes import Reservoir, Ridge
from tqdm import tqdm

""" 
Normalizes the input data 'U_train', 'U_test' with the mean and standard deviation of the training input 'U_train' and
the output data 'Y_train', 'Y_test' with the mean and standard deviation of the training output 'Y_train'.
"""
def normalize_data(U_train, Y_train, U_test, Y_test):
    mean_U = np.mean(U_train, axis=0)
    std_U = np.std(U_train, axis=0)
    mean_Y = np.mean(Y_train, axis=0)
    std_Y = np.std(Y_train, axis=0)
    
    U_train = (U_train - mean_U) / std_U
    Y_train = (Y_train - mean_Y) / std_Y
    U_test = (U_test - mean_U) / std_U
    Y_test = (Y_test - mean_Y) / std_Y
    
    return U_train, Y_train, U_test, Y_test

""" 
Calculates the spectral radius of the matrix 'W'.
"""
def spectral_radius(W):
    return max(abs(np.linalg.eigvals(W)))
    
"""
Calculates the root mean squared error between the true values 'Y_test' and the predictions 'Y_pred' normalized by the standard deviation
of 'Y_test'.
(Workaround for 'reservoirpy.observables.nrmse' which uses a depreceated numpy function.)    
For multi-dimensional data, the NRMSEs are first calculated per component and then averaged.
"""
def nrmse(Y_test, Y_pred):
    return np.mean(rmse(Y_test, Y_pred) / np.std(Y_test, axis=0))

""" 
Returns an updated version of the weight matrix 'W' according to anti-Oja's rule.
'x_pre' and 'x_post' are the activations of the pre- and postsynaptic reservoir neurons and 'eta' is the learning rate.
"""
def anti_oja(W, x_pre, x_post, eta):
    x_pre = x_pre.reshape(-1, 1)
    x_post = x_post.reshape(-1, 1)
    if isinstance(W, scipy.sparse.csr_matrix):
        W = W.toarray()
    delta_W = eta * (np.outer(x_post, x_pre) - x_post**2 * W)
    W -= delta_W
    return W

""" 
Returns an updated version of the weight matrix 'W' according to the normalized Anti-Hebbian rule.
'x_pre' and 'x_post' are the activations of the pre- and postsynaptic reservoir neurons and 'eta' is the learning rate.
"""
def normalized_anti_hebbian(W, x_pre, x_post, eta):
    x_pre = x_pre.reshape(-1, 1)
    x_post = x_post.reshape(-1, 1)
    if isinstance(W, scipy.sparse.csr_matrix):
        W = W.toarray()
    delta_W = eta * np.outer(x_post, x_pre)
    W -= delta_W
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    W = np.where(norms == 0, W, W / norms)
    return W

""" 
Returns an updated version of the weight matrix 'W' according to the Bienenstock-Cooper-Munro rule.
'x_pre' and 'x_post' are the activations of the pre- and postsynaptic reservoir neurons and 'eta' is the learning rate.
"""
def bcm(W, x_pre, x_post, eta):
    x_pre = x_pre.reshape(-1, 1)
    x_post = x_post.reshape(-1, 1)
    if isinstance(W, scipy.sparse.csr_matrix):
        W = W.toarray()
    bcm.x_post_list.append(x_post)
    if len(bcm.x_post_list) > 10:
        bcm.x_post_list.pop(0)
    x_post_array = np.array(bcm.x_post_list)
    theta_LTP = np.mean(x_post_array**2, axis=0)
    delta_W = eta * (x_post - theta_LTP) * np.outer(x_post, x_pre)
    W += delta_W
    return W

""" 
Returns an updated version of the weight matrix 'W' according to the Dual-Threshold Bienenstock-Cooper-Munro rule.
'x_pre' and 'x_post' are the activations of the pre- and postsynaptic reservoir neurons, 'eta' is the learning rate, and
'rho' the scaling factor for the LTD threshold.
"""
def dtbcm(W, x_pre, x_post, eta, rho):
    x_pre = x_pre.reshape(-1, 1)
    x_post = x_post.reshape(-1, 1)
    if isinstance(W, scipy.sparse.csr_matrix):
        W = W.toarray()
    dtbcm.x_post_list.append(x_post)
    if len(dtbcm.x_post_list) > 10:
        dtbcm.x_post_list.pop(0)
    x_post_array = np.array(dtbcm.x_post_list)
    theta_LTP = np.mean(x_post_array**2, axis=0)
    theta_LTD = rho * np.sum(x_post_array, axis=0)
    update_mask = x_post > theta_LTD
    delta_W = update_mask * eta * (x_post - theta_LTP) * np.outer(x_post, x_pre)
    W += delta_W
    return W

"""
Updates 'reservoir' by iterating 'epochs'-times over the training data 'U_train' and applying the specified synaptic plasticity 'rule' 
('anti_oja', 'normalized_anti_hebbian', 'bcm', or 'dtbcm'). 
'params' is a dictionary containing the parameters to be passed to the synaptic plasticity rule.
If 'return_sr_list=True', a list containing the spectral radius for each epoch is returned.
"""            
def pretrain_SP(reservoir, U_train, return_sr_list, epochs, rule, params):
    x_pre = reservoir.zero_state()
    rule.x_post_list = [] # only necessary for the BCM and DTBCM rules
    if return_sr_list:
        sr_list = [reservoir.sr] 
    for _ in range(epochs):        
        for t in range(len(U_train)):
            x_post = reservoir.run(U_train[t])
            reservoir.W = rule(reservoir.W, x_pre, x_post, **params)
            x_pre = x_post
        if return_sr_list:
            sr_list.append(spectral_radius(reservoir.W))    
    if return_sr_list:
        return sr_list
    
""" 
Returns an updated version of the reservoir neuron activation function according to the intrinsic plasticity rule.
'x_post' are the activations of the postsynaptic reservoir neurons, 'eta' is the learning rate, and
'mu' and 'sigma' are mean and standard deviation of the targeted Gaussian output distribution.
"""           
def IP(z_post, x_post, eta, mu, sigma):
    x_post = x_post.reshape(-1, 1)
    delta_b = -eta * (-mu/(sigma**2) + x_post/(sigma**2) * (2*sigma**2 + 1 - x_post**2 + mu * x_post))
    delta_a = eta/IP.a + delta_b * z_post
    IP.a += delta_a
    IP.b += delta_b
    return lambda z: np.tanh(IP.a * z + IP.b)

"""
Updates 'reservoir' by iterating 'epochs'-times over the training data 'U_train' and applying the intrinsic plasticity rule.
'params' is a dictionary containing the parameters to be passed to the intrinsic plasticity rule.
If 'return_sr_list=True', a list containing the spectral radius for each epoch is returned.
"""    
def pretrain_IP(reservoir, U_train, return_sr_list, epochs, params):
    x_pre = reservoir.zero_state()
    IP.a = np.ones(reservoir.units).reshape(-1, 1)
    IP.b = 0
    if return_sr_list:
        sr_list = [reservoir.sr]
    for _ in range(epochs):
        for t in range(len(U_train)):
            x_post = reservoir.run(U_train[t])
            z_post = reservoir.Win @ U_train[t].reshape(-1, 1) + reservoir.W @ x_pre.reshape(-1, 1)
            reservoir.activation = IP(z_post, x_post, **params)
            x_pre = x_post
        if return_sr_list:
            sr_list.append(spectral_radius(IP.a * np.eye(reservoir.units) @ reservoir.W))
    if return_sr_list:
        return sr_list
    
""" 
Calculates the optimal regularization parameter alpha with Generalized Cross-Validation. 
'X' is the input data and 'Y' are the target outputs. 'alpha_list' is the list of regularization parameters to test.
"""    
def gcv(X, Y, alpha_list=np.logspace(-6, 6, 100)):
    def V(X, Y, alpha):
        n, N = X.shape
        A = X @ np.linalg.inv(X.T @ X + n * alpha * np.eye(N)) @ X.T
        
        numerator = np.mean((Y - A @ Y)**2)
        denominator = (1 - np.trace(A)/n)**2
        
        return numerator / denominator
        
    return min(alpha_list, key=lambda alpha: V(X, Y, alpha))

""" 
Calculates the mean NRMSE over 'n_runs' runs together with its standard deviation and a list of the NRMSEs from the individual runs. 
The result is saved in a JSON file 'output_file' if a file name is provided. Additionally, the result is returned.
If 'method_name' is provided, the name of the evaluated method is incorporated into the printed message.
'reservoir_params' is a dictionary containing the parameters for initializing the reservoir. 
The ridge regularization parameter is chosen with GCV.
A seed is set at the beginning to make the result of this function reproducible. This increases the comparability of the results returned
by multiple calls of this function, e.g, for different pretraining methods. 
'U_train', 'Y_train', 'U_test', 'Y_test' is the training and testing input and output respectively. 'warmup' is the number of steps that
are discarded at the beginning of the training.
'pretrain_SP_conf' and 'pretrain_IP_conf' are dictionaries which contain the parameters for calling 'pretrain_SP' or 'pretrain_IP'
respectively if either of these pretraining methods should be applied.
If 'return_sr_list=True', the result also includes lists of the average spectral radii and their standard deviations over the epochs of
the pretraining method. Setting this flag only makes sense if either synaptic or intrinsic plasticity is used for pretraining. 
"""
def avg_nrmse(n_runs, reservoir_params, U_train, Y_train, U_test, Y_test, warmup,
              pretrain_SP_conf=None, pretrain_IP_conf=None, return_sr_list=False,
              output_file="", method_name=""):
    rpy.set_seed(42)
    nrmse_list = []
    sr_list = []
    for _ in tqdm(range(n_runs)):
        reservoir = Reservoir(**reservoir_params, input_bias=False)
        if pretrain_SP_conf is not None:
            sr_list.append(pretrain_SP(reservoir, U_train, return_sr_list, **pretrain_SP_conf)) 
        if pretrain_IP_conf is not None:
            sr_list.append(pretrain_IP(reservoir, U_train, return_sr_list, **pretrain_IP_conf))
            
        states_train = reservoir.run(U_train)
        ridge = gcv(states_train, Y_train) 
        readout = Ridge(ridge=ridge)

        esn = reservoir >> readout
        esn = esn.fit(X=U_train, Y=Y_train, warmup=warmup)
        
        Y_pred = esn.run(U_test)
        
        nrmse_list.append(nrmse(Y_test, Y_pred))
        
    if method_name != "":
        method_name = "of " + method_name
    print(f"NRMSE {method_name} averaged over {n_runs} runs: {np.mean(nrmse_list):.5f} (+/- {np.std(nrmse_list):.5f})")
    result_esn = {"avg_nrmse": np.mean(nrmse_list), "nrmse_std": np.std(nrmse_list), "nrmse_list": nrmse_list}
    
    if return_sr_list:
        sr_list = np.array(sr_list)
        avg_sr_list = list(np.mean(sr_list, axis=0))
        result_esn["avg_sr_list"] = avg_sr_list
        sr_std_list = list(np.std(sr_list, axis=0))
        result_esn["sr_std_list"] = sr_std_list
    
    if output_file != "": 
        with open(output_file, "w") as json_file:
            json.dump(result_esn, json_file, indent=4)
    return result_esn

""" 
Calls 'avg_nrmse' for all combinations '(pretrain_SP_conf, pretrain_IP_conf)' provided in 'pretrain_conf_list'.
"""
def evaluate_all(n_runs, reservoir_params, U_train, Y_train, U_test, Y_test, warmup, pretrain_conf_list, return_sr_list=False,
                 output_file_list=[], method_name_list=[]):
    for i, (pretrain_SP_conf, pretrain_IP_conf) in enumerate(pretrain_conf_list):
        avg_nrmse(n_runs=n_runs, reservoir_params=reservoir_params, 
                  U_train=U_train, Y_train=Y_train, U_test=U_test, Y_test=Y_test, warmup=warmup,
                  pretrain_SP_conf=pretrain_SP_conf, pretrain_IP_conf=pretrain_IP_conf, return_sr_list=return_sr_list, 
                  output_file=output_file_list[i], method_name=method_name_list[i])
        
""" 
Calls 'scipy.stats.mannwhitneyu' to perform the Mann-Whitney U test on the lists of errors 'error_list_1', 'error_list_2'
of two methods and interprets the returned p-value given the significance level 'alpha'.
If 'method_names' is provided, the names of the two compared methods are incorporated into the printed message.
"""
def evaluate_mannwhitneyu(error_list_1, error_list_2, alpha, alternative, method_names=["method 1", "method 2"]):
    _, p_value = scipy.stats.mannwhitneyu(error_list_1, error_list_2, alternative=alternative)

    if alternative == "two-sided":
        comparison = "different"
    elif alternative == "less":
        comparison = "smaller"
    else:
        comparison = "larger"
            
    if p_value < alpha:
       print(f"Reject the null hypothesis: Errors of {method_names[0]} are {comparison} than errors of {method_names[1]}.")
    else:
        print(f"Fail to reject the null hypothesis: No evidence that errors of {method_names[0]} are {comparison} than errors of {method_names[1]}.")
        
"""
Plots the NRMSE against the spectral radius which is increased in steps of 0.05 from 0.05 to 1. 
For each spectral radius, NRMSE is averaged over 'n_runs' runs. 
'reservoir_params' is a dictionary containing the parameters for initializing the reservoir.
The ridge regularization parameter is chosen with GCV.
'U_train', 'Y_train', 'U_test', 'Y_test' is the training and testing input and output respectively. 'warmup' is the number of steps that
are discarded at the beginning of the training.
The plot is saved in the directory at 'output_path'. Additionally, the average NRMSE and its standard deviation are saved in a JSON file in
the same directory.
"""
def plot_nrmse_vs_sr(n_runs, reservoir_params, U_train, Y_train, U_test, Y_test, warmup, output_path):
    os.makedirs(output_path, exist_ok=True)

    nrmse_vs_sr_dict = {}
    avg_nrmse_list = []
    nrmse_std_list = []
    for sr in np.arange(0.05, 1.15, 0.05):
        reservoir_params["sr"] = sr
        result_esn = avg_nrmse(n_runs=n_runs, reservoir_params=reservoir_params,
                               U_train=U_train, Y_train=Y_train, U_test=U_test, Y_test=Y_test, warmup=warmup,
                               method_name=f"ESN with sr = {sr:.2f}")
        avg_nrmse_list.append(result_esn["avg_nrmse"])
        nrmse_std_list.append(result_esn["nrmse_std"])
        nrmse_vs_sr_dict[sr] = {"avg_nrmse": result_esn["avg_nrmse"], "nrmse_std": result_esn["nrmse_std"]}

    with open(os.path.join(output_path, "nrmse_vs_sr.json"), "w") as json_file:
        json.dump(nrmse_vs_sr_dict, json_file, indent=4)
        
    plt.errorbar(np.arange(0.05, 1.15, 0.05), avg_nrmse_list, yerr=nrmse_std_list, fmt='o', capsize=5)

    plt.xlabel("spectral radius", fontsize=14)
    plt.xticks(np.arange(0.1, 1.2, 0.1))
    plt.tick_params(labelsize=12)
    plt.ylabel("NRMSE", fontsize=14)    
    plt.title("NRMSE vs. Spectral Radius", fontsize=16)

    plt.savefig(os.path.join(output_path, "nrmse_vs_sr.png"), dpi=500)
    plt.close()

"""
Plots the spectral radius against the pretraining epochs and saves the result as 'output_file'.
'results_dict_SP' and 'results_dict_IP' are dictionaries with the names of the different synaptic and intrinsic pretraining methods 
as keys and dictionaries containing lists 'avg_sr_list' and 'sr_std_list' as entries. These lists contain contain the average spectral 
radius and its standard deviation for each epoch and are created by 'avg_nrmse' if 'return_sr_list=True'.
The optimal spectral radius 'sr_opt' is also marked in the plot.
"""
def plot_sr_vs_epoch(results_dict_SP, results_dict_IP, sr_opt, output_file):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for (method_name, result), color in zip(results_dict_SP.items(), ["tab:green", "tab:red", "blue", "purple"]):
        ax[0].errorbar(range(0, len(result["avg_sr_list"])), result["avg_sr_list"], yerr=result["sr_std_list"], 
                       fmt='o', capsize=5, label=method_name, color=color)
    for (method_name, result), color in zip(results_dict_IP.items(), ["magenta"]):
        ax[1].errorbar(range(0, len(result["avg_sr_list"])), result["avg_sr_list"], yerr=result["sr_std_list"], 
                       fmt='o', capsize=5, label=method_name, color=color)
    
    for axis in ax:
        axis.axhline(y=sr_opt, linestyle='--', label='optimal sr', color="tab:orange")
        axis.tick_params(labelsize=14)
        axis.set_xlabel("epoch", fontsize=16)
        axis.set_ylabel("spectral radius", fontsize=16)
    ax[0].legend(loc="center left", fontsize=14)
    ax[1].legend(fontsize=14)
   
    fig.suptitle("Spectral Radius vs. Epoch", fontsize=22)
    ax[0].set_title("Synaptic Plasticity", fontsize=18)
    ax[1].set_title("Intrinsic Plasticity", fontsize=18)

    plt.tight_layout()
    plt.savefig(output_file, dpi=1000)
    plt.close()
    
