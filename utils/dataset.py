import math
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data_processing.hoa_binh_data_processing import HoaBinhSample, SpectralSample
from utils.data_utils import DataSample, get_avg_value
from utils.consts import CHEMICAL_SUBSTANCE_COLUMNS


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
