import math
import numpy as np
import pandas as pd
from typing import List, Dict
from torch.utils.data import Dataset
import torch
from utils.data_utils import DataSample, get_avg_value
from utils.consts import CHEMICAL_SUBSTANCE_COLUMNS


class ChemicalDataset(Dataset):
    """
    Dataset for SARIMAX model. Returns data in a Pandas DataFrame format.
    """

    def __init__(self, samples: List[DataSample]):
        """
        :param samples: List of data samples.
        """
        self.samples = [sample for sample in samples if not math.isnan(sample.target_value)]

        # Replace NaN values with the average for each chemical substance
        self.nan_default_replace_value = {
            substance: get_avg_value(substance=substance, samples=self.samples)
            for substance in CHEMICAL_SUBSTANCE_COLUMNS
        }
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.tensor(
            [sample.chem_substance_concentration[substance]
             if not math.isnan(sample.chem_substance_concentration[substance]) else
             self.nan_default_replace_value[substance] for substance in CHEMICAL_SUBSTANCE_COLUMNS],
            dtype=torch.float32)
        target = torch.tensor(sample.target_value, dtype=torch.float32)
        return features, target

def dataset_to_dataframe(data_by_location: Dict[str, List[DataSample]]) -> pd.DataFrame:
    """
    Chuyển toàn bộ dữ liệu từ dictionary thành dataframe.
    """
    all_samples = [sample for location in sorted(list(data_by_location)) for sample in data_by_location[location]]
    dataset = ChemicalDataset(all_samples)
    
    data = []
    for sample in dataset.samples:
        row = {
            "year": sample.year,
            "month": sample.month,
            "target_value": sample.target_value
        }
        row.update(sample.chem_substance_concentration)  # Thêm các chất hóa học
        data.append(row)
    
    return pd.DataFrame(data)
