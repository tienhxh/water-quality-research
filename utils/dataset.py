import math
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data_processing.hoa_binh_data_processing import HoaBinhSample, SpectralSample
from utils.data_utils import DataSample, get_avg_value
from utils.consts import CHEMICAL_SUBSTANCE_COLUMNS
from utils.new_data_utils import NewDataSample, get_new_avg_value, NEW_CHEMICAL_SUBSTANCE_COLUMNS


class ChemicalDataset(Dataset):
    """
    A simple dataset.
    """

    def __init__(self, samples: List[DataSample]):
        samples = [sample for sample in samples if not math.isnan(sample.target_value)]
        self.samples = samples
        # We replace the nan value by this value.
        self.nan_default_replace_value = {substance: get_avg_value(substance=substance, samples=self.samples)
                                          for substance in CHEMICAL_SUBSTANCE_COLUMNS}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.tensor(
            [sample.chem_substance_concentration[substance]
             if not math.isnan(sample.chem_substance_concentration[substance]) else
             self.nan_default_replace_value[substance] for substance in CHEMICAL_SUBSTANCE_COLUMNS],
            dtype=torch.float32)
        target = torch.tensor([sample.target_value], dtype=torch.float32)
        return features, target


def get_simple_dataloader(data_by_location: Dict[str, List[DataSample]], batch_size: int = 512,
                          shuffle: bool = False) -> DataLoader:
    """
    Get simple dataloader by combining data from all locations.
    :param data_by_location: Dictionary with the key being the location that the sample was taken and the value is list
    of samples taken from that location.
    :param batch_size: Batch size.
    :param shuffle: Whether to shuffle data.
    :return: Dataloader.
    """
    all_samples = [sample for location in sorted(list(data_by_location)) for sample in data_by_location[location]]
    dataset = ChemicalDataset(all_samples)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class HoabinhDataset1D(Dataset):
    def __init__(self, samples: List[HoaBinhSample], spectral_field: str = "rw",
                 target_field: str = "chla", max_length: int = 500):
        """
        Simple dataset for Hoa Binh lake data using a single spectral component.

        :param samples: List of water samples from Hoa Binh lake.
        :param spectral_field: Spectral component to use.
        :param target_field: Target field.
        :param max_length: Each sample contains data at different wave length. This parameter limit the numbers of wave
        lengths.
        """
        self.samples = samples
        self.spectral_field = spectral_field
        self.target_field = target_field
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # TODO: Calculate the missing value.
        features = torch.tensor(
            [getattr(spectral_sample, self.spectral_field) or 0 for spectral_sample in
             sample.spectral_data[:self.max_length]],
            dtype=torch.float32)
        target = torch.tensor(getattr(sample.chemical_data, self.target_field), dtype=torch.float32)
        return features, target


def get_hoabinh_1D_dataloader(samples: list[HoaBinhSample], spectral_field: str = "rw", target_field: str = "chla",
                              input_size: int = 500, batch_size: int = 4, shuffle: bool = False):
    dataset = HoabinhDataset1D(samples, spectral_field, target_field, input_size)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class NewChemicalDataset(Dataset):
    """
    Dataset chuẩn hóa các biến đầu vào.
    """

    def __init__(self, samples: List[NewDataSample]):
        # Bỏ các sample có target_value bị NaN
        samples = [sample for sample in samples if not math.isnan(sample.target_value)]
        self.samples = samples

        # Giá trị mặc định thay thế NaN
        self.nan_default_replace_value = {
            substance: get_new_avg_value(substance=substance, samples=self.samples)
            for substance in NEW_CHEMICAL_SUBSTANCE_COLUMNS
        }

        # Tính mean và std cho từng chất
        self.mean_std = {}
        for substance in NEW_CHEMICAL_SUBSTANCE_COLUMNS:
            values = [
                sample.chem_substance_concentration[substance]
                if not math.isnan(sample.chem_substance_concentration[substance])
                else self.nan_default_replace_value[substance]
                for sample in self.samples
            ]
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            self.mean_std[substance] = (mean, std if std != 0 else 1.0)  

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = []

        for substance in NEW_CHEMICAL_SUBSTANCE_COLUMNS:
            value = sample.chem_substance_concentration[substance]
            if math.isnan(value):
                value = self.nan_default_replace_value[substance]

            mean, std = self.mean_std[substance]
            normalized_value = (value - mean) / std
            features.append(normalized_value)

        features = torch.tensor(features, dtype=torch.float32)
        target = torch.tensor([sample.target_value], dtype=torch.float32)
        return features, target



def get_new_simple_dataloader(data: List[NewDataSample], batch_size: int = 512,
                          shuffle: bool = False) -> DataLoader:
    """
    Get simple dataloader by combining data from all locations.
    :param data_by_location: Dictionary with the key being the location that the sample was taken and the value is list
    of samples taken from that location.
    :param batch_size: Batch size.
    :param shuffle: Whether to shuffle data.
    :return: Dataloader.
    """
    dataset = NewChemicalDataset(data)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class NewChemicalSequenceDataset(Dataset):
    """
    Dataset for LSTM: Returns sequences with a fixed length.
    """

    def __init__(self, samples: List[NewDataSample], seq_length: int):
        # Filter out all samples with NaN as target.
        samples = [sample for sample in samples if not math.isnan(sample.target_value)]
        self.samples = samples
        self.seq_length = seq_length  # Sequence length.

        # Replace NaN values in chemical concentrations with the corresponding average value.
        self.nan_default_replace_value = {
            substance: get_avg_value(substance=substance, samples=self.samples)
            for substance in NEW_CHEMICAL_SUBSTANCE_COLUMNS
        }
            # Ensure there are enough samples to create at least one full sequence.
        if len(self.samples) >= self.seq_length:
            # Create indices for non-overlapping sequences.
            self.indices = [0] + list(range(seq_length - 1, len(self.samples) - self.seq_length + 1, self.seq_length))
        else:
            # If not enough samples, set indices to an empty list.
            self.indices = []
    def __len__(self):
        # Total number of non-overlapping sequences available.
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the starting index for the current sequence.
        start_idx = self.indices[idx]
        
        if start_idx == 0:
            padding_sample = self.samples[0]  # Lấy giá trị tháng đầu tiên làm padding
            sequence_samples = [padding_sample] + self.samples[start_idx: start_idx + self.seq_length]
        else:
            # Extract a sequence of samples starting from `start_idx`.
            sequence_samples = self.samples[start_idx - 1: start_idx + self.seq_length]

        # Extract features for each sample in the sequence.
        features = torch.stack([
            torch.tensor(
                [
                    sample.chem_substance_concentration[substance] 
                    if not math.isnan(sample.chem_substance_concentration[substance])
                    else self.nan_default_replace_value[substance]
                    for substance in NEW_CHEMICAL_SUBSTANCE_COLUMNS
                ],
                dtype=torch.float32
            ) for sample in sequence_samples
        ])

        # Extract target values for the entire sequence.
        targets = torch.tensor(
            [sample.target_value for sample in sequence_samples], 
            dtype=torch.float32
        )  # Shape: (seq_length,).
        
        return features, targets


def get_new_lstm_dataloader(data: List[DataSample], 
                        seq_length: int, batch_size: int = 4, 
                        shuffle: bool = False) -> DataLoader:
    """
    Create a DataLoader for LSTM.
    :param data_by_location: Dictionary with key being the location and value being the samples.
    :param batch_size: Batch size.
    :param seq_length: Sequence length.
    :param shuffle: Whether to shuffle the data.
    :return: Dataloader.
    """
    # Create a dataset with a fixed sequence length.
    dataset = NewChemicalSequenceDataset(data, seq_length)

    # Create a DataLoader.
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader 