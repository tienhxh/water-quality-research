from typing import List, Dict

import pandas as pd

from utils.data_utils import DataSample
from utils.dataset import ChemicalDataset


def dataset_to_dataframe(data_by_location: Dict[str, List[DataSample]]) -> pd.DataFrame:
    """
    Convert all data from Datasample to a Pandas DataFrame.
    :param data_by_location: Dictionary with the key being the location that the sample was taken and the value is list
    """
    all_samples = [sample for location in sorted(list(data_by_location)) for sample in data_by_location[location]]
    dataset = ChemicalDataset(all_samples)
    
    data = []
    for sample in dataset.samples:
        row = {
            "target_value": sample.target_value
        }
        row.update(sample.chem_substance_concentration)  
        data.append(row)
    
    return pd.DataFrame(data)
