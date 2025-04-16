import math
import statistics
from bisect import bisect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

import numpy as np
import pandas as pd
from tqdm import tqdm

NEW_CHEMICAL_SUBSTANCE_COLUMNS = [
    'LAKE ELEVATION (m)',
    'LAKE VOLUME (m^3)',
    'SALINITY (%)',
    'TEMPERATURE (C)', 
    'SECCHI DEPTH (m)'
]

class NewDataSample(NamedTuple):
    """
    """
    year: int
    month: int
    chem_substance_concentration: Dict[str, Any]
    target_value: float


def month_str_to_number(month_str: str) -> int:
    """
    Chuyển tháng viết tắt (Jan, Feb, ...) sang số (1, 2, ...).

    :param month_str: Tên tháng viết tắt (ví dụ: 'Jan', 'Feb').
    :return: Số tháng tương ứng (1-12).
    """
    from datetime import datetime
    return datetime.strptime(month_str.strip().capitalize(), '%b').month


def load_data() -> List[NewDataSample]:
    """
    Read data from excel files.
    :param files: Excel files that contain the data.
    :param data_dir: Folder that contains the data files.
    :return: Dataframe.
    """
    file_path = 'data//25692_2_data_set_270203_lfqr22.csv'
    df = pd.read_csv(file_path, thousands=',', skipinitialspace=True)
    df.columns = df.columns.str.strip()
    full_data: List[NewDataSample] = []
    for index, row in tqdm(df.iterrows()):
        year = row["YEAR"]
        month = month_str_to_number(row["MONTH"])

        chem_substance_concentration = {chemical_substance: row[chemical_substance] for chemical_substance in
                                        NEW_CHEMICAL_SUBSTANCE_COLUMNS}
        target = row["CHLa (ug/l)"]

        full_data.append(NewDataSample(year=year, month=month,
                                    chem_substance_concentration=chem_substance_concentration, target_value=target))
    return full_data


def split_train_test(full_data: Dict[str, List[NewDataSample]],
                     start_test_year: int = 2004) -> Tuple[List[NewDataSample], List[NewDataSample]]:
    """
    Split the full data into training and testing dataset. Due to the nature of the data, the testing set is selected
    from the tail.
    :param full_data: Full data.
    :param start_test_year: The starting year of testing set. All samples from this year are considered as testing set
    and all samples from the previous years are considered as training set.
    :return: Training and testing dataset.
    """
    training_data: List[NewDataSample] = {}
    testing_data: List[NewDataSample] = {}
    start_test_index = bisect([sample.year for sample in full_data], start_test_year)
    print(f"Split data into {start_test_index} training samples and {len(full_data) - start_test_index} testing samples")

    training_data = full_data[:start_test_index]
    testing_data = full_data[start_test_index:]
    return training_data, testing_data


def get_new_avg_value(substance: str, samples: List[NewDataSample]) -> float:
    """
    Get default value for a substance.
    """
    values = [sample.chem_substance_concentration[substance] for sample in samples]
    filtered_data = [value for value in values if not math.isnan(value)]
    return statistics.mean(filtered_data)
