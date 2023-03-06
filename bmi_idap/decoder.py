import numpy as np
import torch

# import bnpm.similarity
# import bnpm.decomposition
# import bnpm.ca2p_preprocessing
# import bnpm.path_helpers
# import bnpm.file_helpers
# import bnpm.linear_regression
# import bnpm.spectral
# import bnpm.timeSeries


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
        self.d_bools = torch.cat([torch.arange(0, self.n_components)[None,:] == ii for ii in range(self.n_components)], dim=0).type(self._dtype).to(self._device)  # shape: (n_components, n_components)

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

        # Compute factor magnitudes
        M = ((X.T @ F) / torch.linalg.norm(F, dim=0)).T  # shape: (n_components, n_timepoints)

        # Compute cursor as (cosine_similarity(D, i) * projection_magnitude(D_i))**p
        # d_bools = torch.cat([torch.arange(0, M.shape[1])[None,:] == ii for ii in range(M.shape[1])], dim=0).type(self._dtype).to(self._device)  # shape: (n_components, n_components)
        CS = torch.cat([torch.nn.functional.cosine_similarity(M, db_ii[:,None], dim=0, eps=1e-8)[None,:] for db_ii in self.d_bools], dim=0)  # shape: (n_components, n_timepoints)
        D = torch.abs(CS)**p * M  # shape: (n_components, n_timepoints)

        return D, CS, M