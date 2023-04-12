import torch
import numpy as np

import bnpm.similarity

def orthogonalize_matrix(arr):
    """
    Orthogonalizes the rows of an array in order.
    """
    means = arr.mean(1)
    arr_orth = arr - means[:,None]
    for ii in range(1, arr.shape[0]):
        arr_orth[ii] = bnpm.similarity.orthogonalize(arr_orth[ii], arr_orth[:ii].T, method='OLS')[0].squeeze()

    return arr_orth + means[:,None]

def find_EVR_neuralData_factors(neural_data, factors, device='cpu'):
    """
    neural_data (torch.Tensor):
        shape: (neurons, time) (but can be transposed)
    factors (torch.Tensor):
        shape: (factor, time) or (factor, neurons)
    """
    out = [bnpm.similarity.orthogonalize(
        torch.as_tensor(neural_data).type(torch.float32).to(device), 
        torch.as_tensor(factors[ii]).type(torch.float32).to(device), 
        method='OLS',
        device=device) for ii in range(len(factors))]

    EVR_total_weighted = np.array([o[2].cpu().numpy() for o in out])
    EVR = np.stack([np.nan_to_num(o[1].cpu().numpy(), 0) for o in out], axis=0).T
    
    return EVR, EVR_total_weighted

def order_factors_by_EVR(data, factors, device='cpu'):
    EVR, EVR_total_weighted = find_EVR_neuralData_factors(data, factors, device=device)
    idx_ordered = np.argsort(EVR_total_weighted)[::-1]
    factors_ordered = factors[idx_ordered]
    
    return factors_ordered, idx_ordered, EVR_total_weighted[idx_ordered]