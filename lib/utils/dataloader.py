from doctest import Example
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import logging, sys
from tqdm import tqdm

class CERN_Dataset(Dataset):

    # TODO: refactor the extract_cern_examples so we only have one for both CNN and FC

    """
    CERN PyTorch Dataset Class

    ...

    Attributes
    ----------
    cern_data : Tensor
        processed CERN data
    cern_data_noisy : Tensor
        noise-added CERN data
    mode : str
        method for which the CERN dataset has been processed ('fc' or 'cnn)

    Methods
    -------
    preprocess_cern_fc()
        Initialization process of dataset for 'fc' fully-connected mode
    preprocess_cern_cnn()
        Initialization process of dataset for 'cnn' convolutional neural network mode
    """

    def __init__(self, folder_path: str ='../../dataset/interim/', mode: str = 'cnn', # Required arguments
                 train: bool = True, no_days: int = 12, reshape_factor: int = 2, # CNN arguments
                 fc_days: int = 1, # FC arguments
                 N: int = None, seed: int = 0, eps: float = 1e-12,  # Other arguments
                 noise_type: str = 'none', noise_pct: float = 0.9): # Noise arguments

        """
        Parameters
        ----------
        folder_path : string
                Directory with all the files (processed)
        mode: string 
                Either 'cnn' for convolutional usage or 'fc' for basic AE usage.
        train : bool
                Determines whether training or test dataset to be used (already preprocessed to save time)
        no_days : int (optional) 
                Number of days (i.e., rows in the matrix example)
        reshape_factor : int (optional) 
                Used by the original authors to achieve a square tensor
        N : int (optional) 
                Select subset of examples, AFTER reshaping.
        seed : int (optional) 
                Set seed, mainly for shuffling
        eps : float (optional) 
                For numerical stability in min-max normalization.
        noise_type: bool (optional) ('gauss', 'speckle', None)
                If, and what type, of noise to be added to dataset
        noise_pct: float (optional)
                Parameter controlling how much noise to add
        """
        
        # Set seed
        torch.manual_seed(seed)
        
        # Activate tqdm for pandas and remember object variables
        tqdm.pandas()
        self.eps = eps
        self.mode = mode
        
        if train:
            file_path = folder_path + 'cern_train.csv'
        else:
            file_path = folder_path + 'cern_test.csv' # Note that this has all been directly preprocessed to reduce time spent reprocessing.

        df = pd.read_csv(file_path)
        
        if mode == 'cnn':
        
            # Perform first reshape into Tensor of shape (no_examples, no_days, 48)
            self.cern_data = self.preprocess_cern_cnn(df, no_days)


            # Perform second reshape into Tensor of shape (no_examples, no_days * reshape_factor, 48 / reshape_factor)        
            self.cern_data = self.cern_data.reshape(self.cern_data.size(0), self.cern_data.size(1) * reshape_factor,
                                                    self.cern_data.size(2) // reshape_factor)

            # Unsqueeze channel 1 back out (1 filter)
            self.cern_data = self.cern_data.unsqueeze(1)
            
        elif mode == 'fc':
            
            self.cern_data = self.preprocess_cern_fc(df, fc_days)
            
        else:
            
            raise ValueError("Mode must be 'cnn' or 'fc'.")
            
        # If N is selected, pick random list
        if N is not None:
            if N > self.cern_data.shape[0]:
                raise ValueError("Cannot exceed dataset size of {}".format(self.cern_data.size(0)))
            else:
                # Permutation
                # perm = torch.randperm(self.cern_data.size(0))[:N]
                # self.cern_data = self.cern_data[perm, :, :]

                self.cern_data = self.cern_data[:N] # for debug purposes # TODO: remove this

        # Add noise to dataset
        if noise_type == 'gauss':
            # Add Gaussian noise
            noise = torch.randn(self.cern_data.size()) * noise_pct
            self.cern_data_noisy = self.cern_data + noise
            # Clamp between 0 and 1 (same as real life)
            self.cern_data_noisy = torch.clamp(self.cern_data_noisy, min = 0, max = 1)
        elif noise_type == 'speckle':
            raise NotImplementedError('Speckle noise not yet implemented')
        elif noise_type == 'none':
            self.cern_data_noisy = self.cern_data
        else:
            raise NotImplementedError('Noise selection has not been implemented.')


    def __len__(self):
        return self.cern_data.size(0)
    
    def preprocess_cern_fc(self, df, fc_days):
        
        return_torch = torch.zeros(1, fc_days * 48)
        
        def extract_cern_examples(subset_df, fc_days):
            
            """
            Nested function; group-by to modify nonlocal variable self.return_torch and attaches all modified examples.
            """
    
            meter_torch = torch.from_numpy(subset_df.kwh.to_numpy()).float() # conver to Tnesor
            meter_torch = meter_torch.reshape(-1, 48) # reshape into days
            
            # Min max normalizes all tensors (will need to save these values realistically)
            min_meter_day, _ = torch.min(meter_torch, dim = 1, keepdim=True)
            max_meter_day, _ = torch.max(meter_torch, dim = 1, keepdim=True)
            meter_torch = (meter_torch - min_meter_day) / (max_meter_day - min_meter_day + self.eps) # numerica stability
            
            meter_torch = meter_torch[:(meter_torch.shape[0] // (fc_days)) * (fc_days), :] # drop extra rows (cannot be used)
            meter_torch = meter_torch.reshape(-1, fc_days * 48) # reshape to daily form

            nonlocal return_torch # binds to non-global variable, which will be in non-nested function
            return_torch = torch.cat((return_torch, meter_torch))
        
        # nb: Below function does not need to be assigned, as effectively modifies return_torch inplace.
        df.groupby('metre_id').progress_apply(extract_cern_examples, fc_days = fc_days)
        
        return return_torch[1:, :] # Removes first row of 0s
    
    def preprocess_cern_cnn(self, df, no_days):
        
        return_torch = torch.zeros(1, 12, 48)
        
        def extract_cern_examples(subset_df, no_days):
            
            """
            Nested function; group-by to modify nonlocal variable self.return_torch and attaches all modified examples.
            """
    
            meter_torch = torch.from_numpy(subset_df.kwh.to_numpy()).float() # conver to Tnesor
            meter_torch = meter_torch.reshape(-1, 1, 48) # reshape into days
            
            # Min max normalizes all tensors (will need to save these values realistically)
            min_meter_day, _ = torch.min(meter_torch, dim = 2, keepdim=True)
            max_meter_day, _ = torch.max(meter_torch, dim = 2, keepdim=True)
            meter_torch = (meter_torch - min_meter_day) / (max_meter_day - min_meter_day + self.eps) # numerica stability
            
            meter_torch = meter_torch[:(meter_torch.shape[0] // no_days) * no_days, :, :] # drop extra rows (cannot be used)
            meter_torch = meter_torch.reshape(-1, no_days, 48) # reshape to 12 day form

            nonlocal return_torch # binds to non-global variable, which will be in non-nested function
            return_torch = torch.cat((return_torch, meter_torch))
        
        # nb: Below function does not need to be assigned, as effectively modifies return_torch inplace.
        df.groupby('metre_id').progress_apply(extract_cern_examples, no_days = no_days)
        
        return return_torch[1:, :, :] # Removes first row of 0s

    def __getitem__(self, idx):
        
        example = self.cern_data[idx]
        noisy_example = self.cern_data_noisy[idx]

        return example, noisy_example


class CERN_Dataset_V2(Dataset):

    """
    CERN PyTorch Dataset Class V2
    Smaller dataset, and better cleaning. Consecutive days only.

    ...

    Attributes
    ----------
    cern_data : Tensor
        processed CERN data
    cern_data_noisy : Tensor
        noise-added CERN data
    mode : str
        method for which the CERN dataset has been processed ('fc' or 'cnn)

    Methods
    -------
    preprocess_cern_fc()
        Initialization process of dataset for 'fc' fully-connected mode
    preprocess_cern_cnn()
        Initialization process of dataset for 'cnn' convolutional neural network mode
    """

    def __init__(self, folder_path: str ='../../dataset/interim/', mode: str = 'cnn', # Required arguments
                 train: bool = True, no_days: int = 12, reshape_factor: int = 2, # CNN arguments
                 N: int = None, seed: int = 0, eps: float = 1e-12,  # Other arguments
                 noise_type: str = 'none', noise_pct: float = 0.9): # Noise arguments

        """
        Parameters
        ----------
        folder_path : string
                Directory with all the files (processed)
        mode: string 
                Either 'cnn' for convolutional usage or 'fc' for basic AE usage.
        train : bool
                Determines whether training or test dataset to be used (already preprocessed to save time)
        no_days : int (optional) 
                Number of days (i.e., rows in the matrix example)
        reshape_factor : int (optional) 
                Used by the original authors to achieve a square tensor
        N : int (optional) 
                Select subset of examples, AFTER reshaping.
        seed : int (optional) 
                Set seed, mainly for shuffling
        eps : float (optional) 
                For numerical stability in min-max normalization.
        noise_type: bool (optional) ('gauss', 'speckle', None)
                If, and what type, of noise to be added to dataset
        noise_pct: float (optional)
                Parameter controlling how much noise to add
        """
        
        # Set seed
        torch.manual_seed(seed)
        
        # Activate tqdm for pandas and remember object variables
        tqdm.pandas()
        self.eps = eps
        self.mode = mode
        
        if train:
            file_path = folder_path + 'cern_train_v2.csv'
        else:
            file_path = folder_path + 'cern_test_v2.csv' # Note that this has all been directly preprocessed to reduce time spent reprocessing.

        df = pd.read_csv(file_path)
        
        if mode == 'cnn':
        
            # Perform first reshape into Tensor of shape (no_examples, no_days, 48)
            self.cern_data = self.preprocess_cern_cnn(df, no_days)


            # Perform second reshape into Tensor of shape (no_examples, no_days * reshape_factor, 48 / reshape_factor)        
            self.cern_data = self.cern_data.reshape(self.cern_data.size(0), self.cern_data.size(1) * reshape_factor,
                                                    self.cern_data.size(2) // reshape_factor)

            # Unsqueeze channel 1 back out (1 filter)
            self.cern_data = self.cern_data.unsqueeze(1)
            
        elif mode == 'fc':
            
            self.cern_data = self.preprocess_cern_fc(df, no_days)
            
        else:
            
            raise ValueError("Mode must be 'cnn' or 'fc'.")
            
        # If N is selected, pick random list
        if N is not None:
            if N > self.cern_data.shape[0]:
                raise ValueError("Cannot exceed dataset size of {}".format(self.cern_data.size(0)))
            else:
                # Permutation
                # perm = torch.randperm(self.cern_data.size(0))[:N]
                # self.cern_data = self.cern_data[perm, :, :]

                self.cern_data = self.cern_data[:N] # for debug purposes # TODO: remove this

        # Add noise to dataset
        if noise_type == 'gauss':
            # Add Gaussian noise
            noise = torch.randn(self.cern_data.size()) * noise_pct
            self.cern_data_noisy = self.cern_data + noise
            # Clamp between 0 and 1 (same as real life)
            self.cern_data_noisy = torch.clamp(self.cern_data_noisy, min = 0, max = 1)
        elif noise_type == 'speckle':
            raise NotImplementedError('Speckle noise not yet implemented')
        elif noise_type == 'none':
            self.cern_data_noisy = self.cern_data
        else:
            raise NotImplementedError('Noise selection has not been implemented.')


    def __len__(self):
        return self.cern_data.size(0)
    
    def preprocess_cern_fc(self, df, fc_days):
        
        return_torch = torch.zeros(1, fc_days * 48)
        
        def extract_cern_examples(subset_df, fc_days):
            
            """
            Nested function; group-by to modify nonlocal variable self.return_torch and attaches all modified examples.
            """
    
            meter_torch = torch.from_numpy(subset_df.kwh.to_numpy()).float() # conver to Tnesor
            meter_torch = meter_torch.reshape(-1, 48) # reshape into days

            assert meter_torch.shape[0] // (fc_days) != 0, "not enough data for required shape"
            meter_torch = meter_torch[:(meter_torch.shape[0] // (fc_days)) * (fc_days), :] # drop extra rows (cannot be used)

            meter_torch = meter_torch.reshape(-1, fc_days * 48) # reshape to daily form

            nonlocal return_torch # binds to non-global variable, which will be in non-nested function
            return_torch = torch.cat((return_torch, meter_torch))
        
        # nb: Below function does not need to be assigned, as effectively modifies return_torch inplace.
        df.groupby('metre_id').progress_apply(extract_cern_examples, fc_days = fc_days)
        
        return return_torch[1:, :] # Removes first row of 0s
    
    def preprocess_cern_cnn(self, df, no_days):
        
        return_torch = torch.zeros(1, no_days, 48)
        
        def extract_cern_examples(subset_df, no_days):
            
            """
            Nested function; group-by to modify nonlocal variable self.return_torch and attaches all modified examples.
            """
    
            meter_torch = torch.from_numpy(subset_df.kwh.to_numpy()).float() # conver to Tnesor
            meter_torch = meter_torch.reshape(-1, 1, 48) # reshape to 12 day form

            assert meter_torch.shape[0] // no_days, "not enough data for required shape"
            meter_torch = meter_torch[:(meter_torch.shape[0] // no_days) * no_days, :, :] # drop extra rows (cannot be used)

            meter_torch = meter_torch.reshape(-1, no_days, 48)

            nonlocal return_torch # binds to non-global variable, which will be in non-nested function
            return_torch = torch.cat((return_torch, meter_torch))
        
        # nb: Below function does not need to be assigned, as effectively modifies return_torch inplace.
        df.groupby('metre_id').progress_apply(extract_cern_examples, no_days = no_days)
        
        return return_torch[1:, :, :] # Removes first row of 0s

    def __getitem__(self, idx):
        
        example = self.cern_data[idx]
        noisy_example = self.cern_data_noisy[idx]

        return example, noisy_example


class CERN_Dataset_V3(Dataset):

    """
    CERN PyTorch Dataset Class V3
    Smaller dataset, and better cleaning. Consecutive days only.
    No normalisation in preprocessing.

    ...

    Attributes
    ----------
    cern_data : Tensor
        processed CERN data
    cern_data_noisy : Tensor
        noise-added CERN data
    mode : str
        method for which the CERN dataset has been processed ('fc' or 'cnn)

    Methods
    -------
    preprocess_cern_fc()
        Initialization process of dataset for 'fc' fully-connected mode
    preprocess_cern_cnn()
        Initialization process of dataset for 'cnn' convolutional neural network mode
    """

    def __init__(self, folder_path: str ='../../dataset/interim/', mode: str = 'cnn', # Required arguments
                 train: bool = True, no_days: int = 12, reshape_factor: int = 2, # CNN arguments
                 N: int = None, seed: int = 0, eps: float = 1e-12,  # Other arguments
                 noise_type: str = 'none', noise_pct: float = 0.9): # Noise arguments

        """
        Parameters
        ----------
        folder_path : string
                Directory with all the files (processed)
        mode: string 
                Either 'cnn' for convolutional usage or 'fc' for basic AE usage.
        train : bool
                Determines whether training or test dataset to be used (already preprocessed to save time)
        no_days : int (optional) 
                Number of days (i.e., rows in the matrix example)
        reshape_factor : int (optional) 
                Used by the original authors to achieve a square tensor
        N : int (optional) 
                Select subset of examples, AFTER reshaping.
        seed : int (optional) 
                Set seed, mainly for shuffling
        eps : float (optional) 
                For numerical stability in min-max normalization.
        noise_type: bool (optional) ('gauss', 'speckle', None)
                If, and what type, of noise to be added to dataset
        noise_pct: float (optional)
                Parameter controlling how much noise to add
        """
        
        # Set seed
        torch.manual_seed(seed)
        
        # Activate tqdm for pandas and remember object variables
        tqdm.pandas()
        self.eps = eps
        self.mode = mode
        
        if train:
            file_path = folder_path + 'cern_train_unnormal.csv'
        else:
            file_path = folder_path + 'cern_test_unnormal.csv'

        df = pd.read_csv(file_path)
        
        if mode == 'cnn':
        
            # Perform first reshape into Tensor of shape (no_examples, no_days, 48)
            self.cern_data = self.preprocess_cern_cnn(df, no_days)


            # Perform second reshape into Tensor of shape (no_examples, no_days * reshape_factor, 48 / reshape_factor)        
            self.cern_data = self.cern_data.reshape(self.cern_data.size(0), self.cern_data.size(1) * reshape_factor,
                                                    self.cern_data.size(2) // reshape_factor)

            # Unsqueeze channel 1 back out (1 filter)
            self.cern_data = self.cern_data.unsqueeze(1)
            
        elif mode == 'fc':
            
            self.cern_data = self.preprocess_cern_fc(df, no_days)
            
        else:
            
            raise ValueError("Mode must be 'cnn' or 'fc'.")
            
        # If N is selected, pick random list
        if N is not None:
            if N > self.cern_data.shape[0]:
                raise ValueError("Cannot exceed dataset size of {}".format(self.cern_data.size(0)))
            else:
                # Permutation
                # perm = torch.randperm(self.cern_data.size(0))[:N]
                # self.cern_data = self.cern_data[perm, :, :]

                self.cern_data = self.cern_data[:N] # for debug purposes # TODO: remove this

        # Add noise to dataset
        if noise_type == 'gauss':
            # Add Gaussian noise
            noise = torch.randn(self.cern_data.size()) * noise_pct
            self.cern_data_noisy = self.cern_data + noise
            # Clamp between 0 and 1 (same as real life)
            self.cern_data_noisy = torch.clamp(self.cern_data_noisy, min = 0, max = 1)
        elif noise_type == 'speckle':
            raise NotImplementedError('Speckle noise not yet implemented')
        elif noise_type == 'none':
            self.cern_data_noisy = self.cern_data
        else:
            raise NotImplementedError('Noise selection has not been implemented.')
            
        # Min max normalise
        if mode == 'cnn':
            min_meter_day = torch.amin(self.cern_data, dim = (-1, -2), keepdim=True)
            max_meter_day = torch.amax(self.cern_data, dim = (-1, -2), keepdim=True)
            self.cern_data = (self.cern_data - min_meter_day) / (max_meter_day - min_meter_day) # numerica stability
            min_meter_day = torch.amin(self.cern_data_noisy, dim = (-1, -2), keepdim=True)
            max_meter_day = torch.amax(self.cern_data_noisy, dim = (-1, -2), keepdim=True)
            self.cern_data_noisy = (self.cern_data_noisy - min_meter_day) / (max_meter_day - min_meter_day) # numerica stability
        elif mode == 'fc':
            min_meter_day = torch.amin(self.cern_data, dim = (-1), keepdim=True)
            max_meter_day = torch.amax(self.cern_data, dim = (-1), keepdim=True)
            self.cern_data = (self.cern_data - min_meter_day) / (max_meter_day - min_meter_day) # numerica stability
            min_meter_day = torch.amin(self.cern_data_noisy, dim = (-1), keepdim=True)
            max_meter_day = torch.amax(self.cern_data_noisy, dim = (-1), keepdim=True)
            self.cern_data_noisy = (self.cern_data_noisy - min_meter_day) / (max_meter_day - min_meter_day) # numerica stability
        else:
            raise NotImplementedError('mode not implemented')

    def __len__(self):
        return self.cern_data.size(0)
    
    def preprocess_cern_fc(self, df, fc_days):
        
        return_torch = torch.zeros(1, fc_days * 48)
        
        def extract_cern_examples(subset_df, fc_days):
            
            """
            Nested function; group-by to modify nonlocal variable self.return_torch and attaches all modified examples.
            """
    
            meter_torch = torch.from_numpy(subset_df.kwh.to_numpy()).float() # conver to Tnesor
            meter_torch = meter_torch.reshape(-1, 48) # reshape into days

            assert meter_torch.shape[0] // (fc_days) != 0, "not enough data for required shape"
            meter_torch = meter_torch[:(meter_torch.shape[0] // (fc_days)) * (fc_days), :] # drop extra rows (cannot be used)

            meter_torch = meter_torch.reshape(-1, fc_days * 48) # reshape to daily form

            nonlocal return_torch # binds to non-global variable, which will be in non-nested function
            return_torch = torch.cat((return_torch, meter_torch))
        
        # nb: Below function does not need to be assigned, as effectively modifies return_torch inplace.
        df.groupby('metre_id').progress_apply(extract_cern_examples, fc_days = fc_days)
        
        return return_torch[1:, :] # Removes first row of 0s
    
    def preprocess_cern_cnn(self, df, no_days):
        
        return_torch = torch.zeros(1, no_days, 48)
        
        def extract_cern_examples(subset_df, no_days):
            
            """
            Nested function; group-by to modify nonlocal variable self.return_torch and attaches all modified examples.
            """
    
            meter_torch = torch.from_numpy(subset_df.kwh.to_numpy()).float() # conver to Tnesor
            meter_torch = meter_torch.reshape(-1, 1, 48) # reshape to 12 day form

            assert meter_torch.shape[0] // no_days, "not enough data for required shape"
            meter_torch = meter_torch[:(meter_torch.shape[0] // no_days) * no_days, :, :] # drop extra rows (cannot be used)

            meter_torch = meter_torch.reshape(-1, no_days, 48)

            nonlocal return_torch # binds to non-global variable, which will be in non-nested function
            return_torch = torch.cat((return_torch, meter_torch))
        
        # nb: Below function does not need to be assigned, as effectively modifies return_torch inplace.
        df.groupby('metre_id').progress_apply(extract_cern_examples, no_days = no_days)
        
        return return_torch[1:, :, :] # Removes first row of 0s
    
    def __getitem__(self, idx):
        
        example = self.cern_data[idx]
        noisy_example = self.cern_data_noisy[idx]

        return example, noisy_example

class UMASS_Dataset(Dataset):

    """
    UMASS PyTorch Dataset Class

    ...

    Attributes
    ----------
    umass_data : Tensor
        processed umass data
    umass_data_noisy : Tensor
        noise-added umass data
    mode : str
        method for which the umass dataset has been processed ('fc' or 'cnn)

    Methods
    -------
    preprocess_umass_fc()
        Initialization process of dataset for 'fc' fully-connected mode
    preprocess_umass_cnn()
        Initialization process of dataset for 'cnn' convolutional neural network mode
    """

    def __init__(self, folder_path: str ='../../dataset/interim/', mode: str = 'cnn', # Required arguments
                 train: bool = True, no_days: int = 12, reshape_factor: int = 2, # CNN arguments
                 N: int = None, seed: int = 0, eps: float = 1e-12,  # Other arguments
                 noise_type: str = 'none', noise_pct: float = 0.9): # Noise arguments

        """
        Parameters
        ----------
        folder_path : string
                Directory with all the files (processed)
        mode: string 
                Either 'cnn' for convolutional usage or 'fc' for basic AE usage.
        train : bool
                Determines whether training or test dataset to be used (already preprocessed to save time)
        no_days : int (optional) 
                Number of days (i.e., rows in the matrix example)
        reshape_factor : int (optional) 
                Used by the original authors to achieve a square tensor
        N : int (optional) 
                Select subset of examples, AFTER reshaping.
        seed : int (optional) 
                Set seed, mainly for shuffling
        eps : float (optional) 
                For numerical stability in min-max normalization.
        noise_type: bool (optional) ('gauss', 'speckle', None)
                If, and what type, of noise to be added to dataset
        noise_pct: float (optional)
                Parameter controlling how much noise to add
        """
        
        # Set seed
        torch.manual_seed(seed)
        
        # Activate tqdm for pandas and remember object variables
        tqdm.pandas()
        self.eps = eps
        self.mode = mode
        
        if train:
            file_path = folder_path + 'umass_train.csv'
        else:
            file_path = folder_path + 'umass_test.csv' # Note that this has all been directly preprocessed to reduce time spent reprocessing.

        df = pd.read_csv(file_path)
        
        if mode == 'cnn':
        
            # Perform first reshape into Tensor of shape (no_examples, no_days, 48)
            self.umass_data = self.preprocess_umass_cnn(df, no_days)


            # Perform second reshape into Tensor of shape (no_examples, no_days * reshape_factor, 48 / reshape_factor)        
            self.umass_data = self.umass_data.reshape(self.umass_data.size(0), self.umass_data.size(1) * reshape_factor,
                                                    self.umass_data.size(2) // reshape_factor)

            # Unsqueeze channel 1 back out (1 filter)
            self.umass_data = self.umass_data.unsqueeze(1)
            
        elif mode == 'fc':
            
            self.umass_data = self.preprocess_umass_fc(df, no_days)
            
        else:
            
            raise ValueError("Mode must be 'cnn' or 'fc'.")
            
        # If N is selected, pick random list
        if N is not None:
            if N > self.umass_data.shape[0]:
                raise ValueError("Cannot exceed dataset size of {}".format(self.umass_data.size(0)))
            else:
                # Permutation
                # perm = torch.randperm(self.umass_data.size(0))[:N]
                # self.umass_data = self.umass_data[perm, :, :]

                self.umass_data = self.umass_data[:N] # for debug purposes # TODO: remove this

        # Add noise to dataset
        if noise_type == 'gauss':
            # Add Gaussian noise
            noise = torch.randn(self.umass_data.size()) * noise_pct
            self.umass_data_noisy = self.umass_data + noise
            # Clamp between 0 and 1 (same as real life)
            self.umass_data_noisy = torch.clamp(self.umass_data_noisy, min = 0, max = 1)
        elif noise_type == 'speckle':
            raise NotImplementedError('Speckle noise not yet implemented')
        elif noise_type == 'none':
            self.umass_data_noisy = self.umass_data
        else:
            raise NotImplementedError('Noise selection has not been implemented.')


    def __len__(self):
        return self.umass_data.size(0)
    
    def preprocess_umass_fc(self, df, fc_days):
        
        return_torch = torch.zeros(1, fc_days * 96)
        
        def extract_umass_examples(subset_df, fc_days):
            
            """
            Nested function; group-by to modify nonlocal variable self.return_torch and attaches all modified examples.
            """
    
            meter_torch = torch.from_numpy(subset_df.kwh.to_numpy()).float() # conver to Tnesor
            meter_torch = meter_torch.reshape(-1, 96) # reshape into days

            assert meter_torch.shape[0] // (fc_days) != 0, "not enough data for required shape"
            meter_torch = meter_torch[:(meter_torch.shape[0] // (fc_days)) * (fc_days), :] # drop extra rows (cannot be used)

            meter_torch = meter_torch.reshape(-1, fc_days * 96) # reshape to daily form

            nonlocal return_torch # binds to non-global variable, which will be in non-nested function
            return_torch = torch.cat((return_torch, meter_torch))
        
        # nb: Below function does not need to be assigned, as effectively modifies return_torch inplace.
        df.groupby('house').progress_apply(extract_umass_examples, fc_days = fc_days)
        
        return return_torch[1:, :] # Removes first row of 0s
    
    def preprocess_umass_cnn(self, df, no_days):
        
        return_torch = torch.zeros(1, no_days, 96)
        
        def extract_umass_examples(subset_df, no_days):
            
            """
            Nested function; group-by to modify nonlocal variable self.return_torch and attaches all modified examples.
            """
    
            meter_torch = torch.from_numpy(subset_df.kwh.to_numpy()).float() # conver to Tnesor
            meter_torch = meter_torch.reshape(-1, 1, 96) # reshape into days

            assert meter_torch.shape[0] // no_days != 0, "not enough data for required shape"
            meter_torch = meter_torch[:(meter_torch.shape[0] // no_days) * no_days, :, :] # drop extra rows (cannot be used)

            meter_torch = meter_torch.reshape(-1, no_days, 96) # reshape to 12 day form

            nonlocal return_torch # binds to non-global variable, which will be in non-nested function
            return_torch = torch.cat((return_torch, meter_torch))
        
        # nb: Below function does not need to be assigned, as effectively modifies return_torch inplace.
        df.groupby('house').progress_apply(extract_umass_examples, no_days = no_days)
        
        return return_torch[1:, :, :] # Removes first row of 0s

    def __getitem__(self, idx):
        
        example = self.umass_data[idx]
        noisy_example = self.umass_data_noisy[idx]

        return example, noisy_example

class UMASS_Dataset_V2(Dataset):

    """
    UMASS PyTorch Dataset Class
    Unnormalised at preprocessing. Normalise here.

    ...

    Attributes
    ----------
    umass_data : Tensor
        processed umass data
    umass_data_noisy : Tensor
        noise-added umass data
    mode : str
        method for which the umass dataset has been processed ('fc' or 'cnn)

    Methods
    -------
    preprocess_umass_fc()
        Initialization process of dataset for 'fc' fully-connected mode
    preprocess_umass_cnn()
        Initialization process of dataset for 'cnn' convolutional neural network mode
    """

    def __init__(self, folder_path: str ='../../dataset/interim/', mode: str = 'cnn', # Required arguments
                 train: bool = True, no_days: int = 12, reshape_factor: int = 2, # CNN arguments
                 N: int = None, seed: int = 0, eps: float = 1e-12,  # Other arguments
                 noise_type: str = 'none', noise_pct: float = 0.9): # Noise arguments

        """
        Parameters
        ----------
        folder_path : string
                Directory with all the files (processed)
        mode: string 
                Either 'cnn' for convolutional usage or 'fc' for basic AE usage.
        train : bool
                Determines whether training or test dataset to be used (already preprocessed to save time)
        no_days : int (optional) 
                Number of days (i.e., rows in the matrix example)
        reshape_factor : int (optional) 
                Used by the original authors to achieve a square tensor
        N : int (optional) 
                Select subset of examples, AFTER reshaping.
        seed : int (optional) 
                Set seed, mainly for shuffling
        eps : float (optional) 
                For numerical stability in min-max normalization.
        noise_type: bool (optional) ('gauss', 'speckle', None)
                If, and what type, of noise to be added to dataset
        noise_pct: float (optional)
                Parameter controlling how much noise to add
        """
        
        # Set seed
        torch.manual_seed(seed)
        
        # Activate tqdm for pandas and remember object variables
        tqdm.pandas()
        self.eps = eps
        self.mode = mode
        
        if train:
            file_path = folder_path + 'umass_train_unnormal.csv'
        else:
            file_path = folder_path + 'umass_test_unnormal.csv' # Note that this has all been directly preprocessed to reduce time spent reprocessing.

        df = pd.read_csv(file_path)
        
        if mode == 'cnn':
        
            # Perform first reshape into Tensor of shape (no_examples, no_days, 48)
            self.umass_data = self.preprocess_umass_cnn(df, no_days)


            # Perform second reshape into Tensor of shape (no_examples, no_days * reshape_factor, 48 / reshape_factor)        
            self.umass_data = self.umass_data.reshape(self.umass_data.size(0), self.umass_data.size(1) * reshape_factor,
                                                    self.umass_data.size(2) // reshape_factor)

            # Unsqueeze channel 1 back out (1 filter)
            self.umass_data = self.umass_data.unsqueeze(1)
            
        elif mode == 'fc':
            
            self.umass_data = self.preprocess_umass_fc(df, no_days)
            
        else:
            
            raise ValueError("Mode must be 'cnn' or 'fc'.")
            
        # If N is selected, pick random list
        if N is not None:
            if N > self.umass_data.shape[0]:
                raise ValueError("Cannot exceed dataset size of {}".format(self.umass_data.size(0)))
            else:
                # Permutation
                # perm = torch.randperm(self.umass_data.size(0))[:N]
                # self.umass_data = self.umass_data[perm, :, :]

                self.umass_data = self.umass_data[:N] # for debug purposes # TODO: remove this
                
        # Add noise to dataset
        if noise_type == 'gauss':
            # Add Gaussian noise
            noise = torch.randn(self.umass_data.size()) * noise_pct
            self.umass_data_noisy = self.umass_data + noise
            # Clamp between 0 and 1 (same as real life)
            self.umass_data_noisy = torch.clamp(self.umass_data_noisy, min = 0, max = 1)
        elif noise_type == 'speckle':
            raise NotImplementedError('Speckle noise not yet implemented')
        elif noise_type == 'none':
            self.umass_data_noisy = self.umass_data
        else:
            raise NotImplementedError('Noise selection has not been implemented.')
            
        # Min max normalise
        if mode == 'cnn':
            min_meter_day = torch.amin(self.umass_data, dim = (-1, -2), keepdim=True)
            max_meter_day = torch.amax(self.umass_data, dim = (-1, -2), keepdim=True)
            self.umass_data = (self.umass_data - min_meter_day) / (max_meter_day - min_meter_day) # numerica stability
            min_meter_day = torch.amin(self.umass_data_noisy, dim = (-1, -2), keepdim=True)
            max_meter_day = torch.amax(self.umass_data_noisy, dim = (-1, -2), keepdim=True)
            self.umass_data_noisy = (self.umass_data_noisy - min_meter_day) / (max_meter_day - min_meter_day) # numerica stability
        elif mode == 'fc':
            min_meter_day = torch.amin(self.umass_data, dim = (-1), keepdim=True)
            max_meter_day = torch.amax(self.umass_data, dim = (-1), keepdim=True)
            self.umass_data = (self.umass_data - min_meter_day) / (max_meter_day - min_meter_day) # numerica stability
            min_meter_day = torch.amin(self.umass_data_noisy, dim = (-1), keepdim=True)
            max_meter_day = torch.amax(self.umass_data_noisy, dim = (-1), keepdim=True)
            self.umass_data_noisy = (self.umass_data_noisy - min_meter_day) / (max_meter_day - min_meter_day) # numerica stability
        else:
            raise NotImplementedError('mode not implemented')


    def __len__(self):
        return self.umass_data.size(0)
    
    def preprocess_umass_fc(self, df, fc_days):
        
        return_torch = torch.zeros(1, fc_days * 96)
        
        def extract_umass_examples(subset_df, fc_days):
            
            """
            Nested function; group-by to modify nonlocal variable self.return_torch and attaches all modified examples.
            """
    
            meter_torch = torch.from_numpy(subset_df.kwh.to_numpy()).float() # conver to Tnesor
            meter_torch = meter_torch.reshape(-1, 96) # reshape into days
            
            assert meter_torch.shape[0] // (fc_days) != 0, "not enough data for required shape"
            meter_torch = meter_torch[:(meter_torch.shape[0] // (fc_days)) * (fc_days), :] # drop extra rows (cannot be used)

            meter_torch = meter_torch.reshape(-1, fc_days * 96) # reshape to daily form

            nonlocal return_torch # binds to non-global variable, which will be in non-nested function
            return_torch = torch.cat((return_torch, meter_torch))
        
        # nb: Below function does not need to be assigned, as effectively modifies return_torch inplace.
        df.groupby('house').progress_apply(extract_umass_examples, fc_days = fc_days)
        
        return return_torch[1:, :] # Removes first row of 0s
    
    def preprocess_umass_cnn(self, df, no_days):
        
        return_torch = torch.zeros(1, no_days, 96)
        
        def extract_umass_examples(subset_df, no_days):
            
            """
            Nested function; group-by to modify nonlocal variable self.return_torch and attaches all modified examples.
            """
    
            meter_torch = torch.from_numpy(subset_df.kwh.to_numpy()).float() # conver to Tnesor
            meter_torch = meter_torch.reshape(-1, 1, 96) # reshape into days
            
            assert meter_torch.shape[0] // no_days != 0, "not enough data for required shape"
            meter_torch = meter_torch[:(meter_torch.shape[0] // no_days) * no_days, :, :] # drop extra rows (cannot be used)

            meter_torch = meter_torch.reshape(-1, no_days, 96) # reshape to 12 day form

            nonlocal return_torch # binds to non-global variable, which will be in non-nested function
            return_torch = torch.cat((return_torch, meter_torch))
        
        # nb: Below function does not need to be assigned, as effectively modifies return_torch inplace.
        df.groupby('house').progress_apply(extract_umass_examples, no_days = no_days)
        
        return return_torch[1:, :, :] # Removes first row of 0s
    

    def __getitem__(self, idx):
        
        example = self.umass_data[idx]
        noisy_example = self.umass_data_noisy[idx]

        return example, noisy_example

class DRED_Dataset(Dataset):

    """
    DRED PyTorch Dataset Class
    Allows selection of "wide_freq" - unlike other datasets which are fixed at daily.
    Required as due to 1 second interval is too large.
    ...

    Attributes
    ----------
    dred_data : Tensor
        processed dred data
    dred_data_noisy : Tensor
        noise-added dred data
    mode : str
        method for which the dred dataset has been processed ('fc' or 'cnn)

    Methods
    -------
    preprocess_dred_fc()
        Initialization process of dataset for 'fc' fully-connected mode
    preprocess_dred_cnn()
        Initialization process of dataset for 'cnn' convolutional neural network mode
    """

    def __init__(self, folder_path: str ='../../dataset/interim/', mode: str = 'cnn', # Required arguments
                 train: bool = True, no_rows: int = 12, reshape_factor: int = 2, # CNN arguments
                 N: int = None, seed: int = 0, eps: float = 1e-12, wide_freq = 60, # Other arguments
                 noise_type: str = 'none', noise_pct: float = 0.9): # Noise arguments

        """
        Parameters
        ----------
        folder_path : string
                Directory with all the files (processed)
        mode: string 
                Either 'cnn' for convolutional usage or 'fc' for basic AE usage.
        train : bool
                Determines whether training or test dataset to be used (already preprocessed to save time)
        no_rows : int (optional) 
                Number of rows (i.e., rows in the matrix example)
        reshape_factor : int (optional) 
                Used by the original authors to achieve a square tensor
        N : int (optional) 
                Select subset of examples, AFTER reshaping.
        seed : int (optional) 
                Set seed, mainly for shuffling
        eps : float (optional) 
                For numerical stability in min-max normalization.
        noise_type: bool (optional) ('gauss', 'speckle', None)
                If, and what type, of noise to be added to dataset
        noise_pct: float (optional)
                Parameter controlling how much noise to add
        """
        
        # Set seed
        torch.manual_seed(seed)
        
        # Activate tqdm for pandas and remember object variables
        tqdm.pandas()
        self.eps = eps
        self.mode = mode
        
        # Within all available frequencies
        self.wide_freq = wide_freq
        
        if train:
            file_path = folder_path + 'dred_train.csv'
        else:
            file_path = folder_path + 'dred_test.csv'

        df = pd.read_csv(file_path)
        
        if mode == 'cnn':
        
            # Perform first reshape into Tensor of shape (no_examples, no_rows, 48)
            self.dred_data = self.preprocess_dred_cnn(df, no_rows)


            # Perform second reshape into Tensor of shape (no_examples, no_rows * reshape_factor, 48 / reshape_factor)        
            self.dred_data = self.dred_data.reshape(self.dred_data.size(0), self.dred_data.size(1) * reshape_factor,
                                                    self.dred_data.size(2) // reshape_factor)

            # Unsqueeze channel 1 back out (1 filter)
            self.dred_data = self.dred_data.unsqueeze(1)
            
        elif mode == 'fc':
            
            self.dred_data = self.preprocess_dred_fc(df, no_rows)
            
        else:
            
            raise ValueError("Mode must be 'cnn' or 'fc'.")
            
        # If N is selected, pick random list
        if N is not None:
            if N > self.dred_data.shape[0]:
                raise ValueError("Cannot exceed dataset size of {}".format(self.dred_data.size(0)))
            else:
                # Permutation
                # perm = torch.randperm(self.dred_data.size(0))[:N]
                # self.dred_data = self.dred_data[perm, :, :]

                self.dred_data = self.dred_data[:N] # for debug purposes # TODO: remove this

        # Add noise to dataset
        if noise_type == 'gauss':
            # Add Gaussian noise
            noise = torch.randn(self.dred_data.size()) * noise_pct
            self.dred_data_noisy = self.dred_data + noise
            # Clamp between 0 and 1 (same as real life)
            self.dred_data_noisy = torch.clamp(self.dred_data_noisy, min = 0, max = 1)
        elif noise_type == 'speckle':
            raise NotImplementedError('Speckle noise not yet implemented')
        elif noise_type == 'none':
            self.dred_data_noisy = self.dred_data
        else:
            raise NotImplementedError('Noise selection has not been implemented.')
            
        if mode == 'cnn':
            min_meter_day = torch.amin(self.dred_data, dim = (-1, -2), keepdim=True)
            max_meter_day = torch.amax(self.dred_data, dim = (-1, -2), keepdim=True)
            self.dred_data = (self.dred_data - min_meter_day) / (max_meter_day - min_meter_day) # numerica stability
            min_meter_day = torch.amin(self.dred_data_noisy, dim = (-1, -2), keepdim=True)
            max_meter_day = torch.amax(self.dred_data_noisy, dim = (-1, -2), keepdim=True)
            self.dred_data_noisy = (self.dred_data_noisy - min_meter_day) / (max_meter_day - min_meter_day) # numerica stability
        elif mode == 'fc':
            min_meter_day = torch.amin(self.dred_data, dim = (-1), keepdim=True)
            max_meter_day = torch.amax(self.dred_data, dim = (-1), keepdim=True)
            self.dred_data = (self.dred_data - min_meter_day) / (max_meter_day - min_meter_day) # numerica stability
            min_meter_day = torch.amin(self.dred_data_noisy, dim = (-1), keepdim=True)
            max_meter_day = torch.amax(self.dred_data_noisy, dim = (-1), keepdim=True)
            self.dred_data_noisy = (self.dred_data_noisy - min_meter_day) / (max_meter_day - min_meter_day) # numerica stability
        else:
            raise NotImplementedError('mode not implemented')        


    def __len__(self):
        return self.dred_data.size(0)
    
    def preprocess_dred_fc(self, df, no_rows):
        
        return_torch = torch.zeros(1, no_rows * self.wide_freq)
        
        def extract_dred_examples(subset_df, no_rows):
            
            """
            Nested function; group-by to modify nonlocal variable self.return_torch and attaches all modified examples.
            """
    
            meter_torch = torch.from_numpy(subset_df.kwh.to_numpy()).float() # convert to Tnesor
            meter_torch = meter_torch.reshape(-1, self.wide_freq) # reshape into days

            assert meter_torch.shape[0] // (no_rows) != 0, "not enough data for required shape"
            meter_torch = meter_torch[:(meter_torch.shape[0] // (no_rows)) * (no_rows), :] # drop extra rows (cannot be used)

            meter_torch = meter_torch.reshape(-1, no_rows * self.wide_freq) # reshape to daily form

            nonlocal return_torch # binds to non-global variable, which will be in non-nested function
            return_torch = torch.cat((return_torch, meter_torch))
        
        # nb: Below function does not need to be assigned, as effectively modifies return_torch inplace.
        extract_dred_examples(df, no_rows = no_rows)
        print('DRED processed.')
        
        return return_torch[1:, :] # Removes first row of 0s
    
    def preprocess_dred_cnn(self, df, no_rows):
        
        return_torch = torch.zeros(1, no_rows, self.wide_freq)
        
        def extract_dred_examples(subset_df, no_rows):
            
            """
            Nested function; group-by to modify nonlocal variable self.return_torch and attaches all modified examples.
            """
    
            meter_torch = torch.from_numpy(subset_df.kwh.to_numpy()).float() # conver to Tnesor
            meter_torch = meter_torch.reshape(-1, 1, self.wide_freq) # reshape to 12 day form

            assert meter_torch.shape[0] // no_rows, "not enough data for required shape"
            meter_torch = meter_torch[:(meter_torch.shape[0] // no_rows) * no_rows, :, :] # drop extra rows (cannot be used)

            meter_torch = meter_torch.reshape(-1, no_rows, self.wide_freq)

            nonlocal return_torch # binds to non-global variable, which will be in non-nested function
            return_torch = torch.cat((return_torch, meter_torch))
        
        # nb: Below function does not need to be assigned, as effectively modifies return_torch inplace.
        extract_dred_examples(df, no_rows = no_rows)
        print('DRED processed.')
        
        return return_torch[1:, :, :] # Removes first row of 0s

    def __getitem__(self, idx):
        
        example = self.dred_data[idx]
        noisy_example = self.dred_data_noisy[idx]

        return example, noisy_example