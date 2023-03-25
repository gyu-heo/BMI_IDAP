import numpy as np
import torch
from tqdm import tqdm

# import bnpm.similarity
# import bnpm.decomposition
# import bnpm.ca2p_preprocessing
# import bnpm.path_helpers
# import bnpm.file_helpers
# import bnpm.linear_regression
# import bnpm.spectral
import bnpm.timeSeries


######################################################################################################################
################################################## FACTOR DEFINITION #################################################
######################################################################################################################

######################################################################################################################
################################################## DECODER DEFINITION ################################################
######################################################################################################################


class Decoder_angle_magnitude:
    def __init__(self, F, power=2.0, dtype=torch.float32, device='cpu'):
        """
        Angle-magnitude decoder.

        Args:
            F (np.ndarray or torch.Tensor):
                Factor matrix.
                Either shape: (n_neurons, n_components) or shape: (n_neurons,)
            p (np.ndarray or torch.Tensor or float or int):
                Power coefficient. Tunes the angular specificity of the decoder.
                Scalar value.
            dtype (torch.dtype):
                Data type to use for computations.
            device (str):
                Device to use for computations. Either 'cpu' or 'cuda'.
        """
        # Set attributes
        self.power = power
        self._dtype = dtype
        self._device = device

        if isinstance(F, np.ndarray):
            self.F = torch.as_tensor(F).type(self._dtype).to(self._device)
        else:
            self.F = F.type(self._dtype).to(self._device)
        assert isinstance(self.F, torch.Tensor), "F must be a torch.Tensor or np.ndarray."

        self.n_components = int(self.F.shape[1])
#         self.d_bools = torch.cat([torch.arange(0, self.n_components)[None,:] == ii for ii in range(self.n_components)], dim=0).type(self._dtype).to(self._device)  # shape: (n_components, n_components)
        self.d_bools = torch.eye(self.n_components, dtype=self._dtype, device=self._device)

    def __call__(self, X, F=None, power=None):
        """
        Angle-magnitude decoder.

        Args:
            X (np.ndarray or torch.Tensor): 
                Neural data (dFoF: either timeseries or timepoint).
                Either shape: (n_neurons, n_timepoints) or shape: (n_neurons,)
            F (np.ndarray or torch.Tensor):
                Factor matrix.
                If None, uses self.F.
                Either shape: (n_neurons, n_components) or shape: (n_neurons,)
            p (np.ndarray or torch.Tensor or float or int):
                Power coefficient. Tunes the angular specificity of the decoder.
                If None, uses self.power.
                Scalar value.

        Returns:
            D (torch.Tensor):
                Values for each decoder dimension.
                Shape: (n_components, n_timepoints)
            CS (torch.Tensor):
                Cosine similarity between each decoder dimension and each timepoint.
                Shape: (n_components, n_timepoints)
            M (torch.Tensor):
                Magnitude of each decoder dimension.
                Shape: (n_components, n_timepoints)
        """
        # Convert to torch.Tensor
        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X).type(self._dtype).to(self._device)
        if F is not None:
            if isinstance(F, np.ndarray):
                F = torch.as_tensor(F).type(self._dtype).to(self._device)
        else:
            F = self.F.type(self._dtype).to(self._device)
        assert F is not None, "F must be provided as an argument or as an attribute."
        
        p = float(power) if power is not None else self.power

        # Check shapes
        if len(X.shape) == 1:
            X = X[:,None]
        if len(F.shape) == 1:
            F = F[:,None]
            
#         X -= torch.mean(X, dim=0)[None,:]

        # Compute factor magnitudes
        # M = ((X.T @ F) / torch.linalg.norm(F, dim=0)).T  # shape: (n_components, n_timepoints)
        M = (torch.nansum(X[:,:,None] * F[:,None,:], dim=0) / torch.linalg.norm(F, dim=0)).T  # shape: (n_components, n_timepoints)
#         M = ((F.T @ X) / torch.linalg.norm(X, dim=0))  # shape: (n_components, n_timepoints)
#         M = torch.linalg.norm(X, dim=0)  # shape: (n_components, n_timepoints)

        # Compute cursor as (cosine_similarity(D, i) * projection_magnitude(D_i))**p
        # d_bools = torch.cat([torch.arange(0, M.shape[1])[None,:] == ii for ii in range(M.shape[1])], dim=0).type(self._dtype).to(self._device)  # shape: (n_components, n_components)
        CS = ((M / torch.linalg.norm(M, dim=0)).T @ (self.d_bools / torch.linalg.norm(self.d_bools, dim=0))).T  # shape: (n_components, n_timepoints)
#         CS = (M.T @ self.d_bools).T / (torch.linalg.norm(M, dim=0)[None,:] * torch.linalg.norm(self.d_bools, dim=0)[:,None])  # shape: (n_components, n_timepoints)
#         CS = torch.cat([torch.nn.functional.cosine_similarity(M, db_ii[:,None], dim=0, eps=1e-8)[None,:] for db_ii in self.d_bools], dim=0)  # shape: (n_components, n_timepoints)
#         CS = (M / torch.linalg.norm(M, dim=0)).T  # shape: (n_components, n_timepoints)

#         CS = ((X / torch.linalg.norm(X, dim=0)).T @ (F / torch.linalg.norm(F, dim=0))).T  # shape: (n_components, n_timepoints)
#         CS = torch.cat([torch.nn.functional.cosine_similarity(X, F[:,ii][:,None], dim=0, eps=1e-8)[None,:] for ii in range(F.shape[1])], dim=0)  # shape: (n_components, n_timepoints)

#         M = CS * torch.linalg.norm(X, dim=0)[None,:]
        
#         D = torch.linalg.norm(M, ord=2, dim=0)

        D = torch.abs(CS)**p * M  # shape: (n_components, n_timepoints)
#         D = CS * M  # shape: (n_components, n_timepoints)

        return D, CS, M
    

# Cursor simulation
def simple_cursor_simulation(
    D, 
    CS,
    M, 
    idx_cursor=0, 
    idx_avg=-1, 
    thresh_reward=1.0, 
#     thresh_quiescence_avgVec=0.9, 
    thresh_quiescence_cursorMag=0.2, 
#     thresh_quiescence_cursor=0.2, 
    thresh_quiescence_cursorDecoder=0.2, 
    duration_quiescence_hold=5, 
    duration_threshold_hold=3, 
    win_smooth_cursor=3
):
    n_samples = D.shape[0]
    sm = {
        'rewards': np.zeros(n_samples, dtype=int),
        'trial_num': np.zeros(n_samples, dtype=int),
        'counter_quiescence': np.zeros(n_samples, dtype=int),
        'counter_threshold': np.zeros(n_samples, dtype=int),
        'CE_trial': np.zeros(n_samples, dtype=int),
        'CS_quiescence': np.zeros(n_samples, dtype=int),
        'cursor': np.zeros(n_samples, dtype=float),
        'timeSeries_avgVec': np.zeros(n_samples, dtype=float),
    }
    
    num_reward = 0
    CE_trial = 1
    counter_quiescence = 0
    counter_threshold = 0
    
    if idx_avg < 0:
        idx_avg = D.shape[1] + idx_avg
        
    kernel = np.concatenate((np.zeros(win_smooth_cursor), np.ones(win_smooth_cursor)))
    kernel = torch.as_tensor(kernel, device=D.device, dtype=D.dtype) if isinstance(D, torch.Tensor) else kernel
    kernel = kernel / kernel.sum()
    D_smooth = bnpm.timeSeries.convolve_torch(D, kernel, padding='same')
    
    
    for ii, (d, cs, m) in tqdm(enumerate(zip(D_smooth, CS, M)), total=len(D_smooth)):
#         CS_quiescence = d[idx_avg] >= thresh_quiescence
#         CS_quiescence = (cs[idx_avg] >= thresh_quiescence_avgVec) * (d[idx_cursor] <= thresh_quiescence_cursor)
        CS_quiescence = (m[idx_cursor] <= thresh_quiescence_cursorMag) * (d[idx_cursor] <= thresh_quiescence_cursorDecoder)
#         CS_quiescence = cs.argmax() == idx_avg
        sm['CS_quiescence'][ii] = CS_quiescence
        
        if CE_trial:
            if d[idx_cursor] > thresh_reward:
                counter_threshold += 1
            else:
                counter_threshold = 0
            if counter_threshold >= duration_threshold_hold:
                CE_trial = 0
                num_reward += 1
                sm['rewards'][ii] = 1
            sm['CE_trial'][ii] = 1
        else:
#             if m <= thresh_quiescence:
            if CS_quiescence:                
                counter_quiescence += 1
            else:
                counter_quiescence = 0
            if counter_quiescence >= duration_quiescence_hold:
                CE_trial = 1
                sm['trial_num'][ii] = sm['trial_num'].max() + 1
                counter_quiescence = 0  
        sm['counter_quiescence'][ii] = counter_quiescence
        sm['counter_threshold'][ii] = counter_threshold
        
        sm['cursor'][ii] = d[idx_cursor].cpu().numpy()
        sm['timeSeries_avgVec'][ii] = cs[idx_avg].cpu().numpy()
                
    return num_reward, sm