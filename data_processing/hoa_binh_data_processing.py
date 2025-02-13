from enum import Enum
from typing import Dict, List, NamedTuple
import pandas as pd
from tqdm import tqdm

from utils.consts import PRJ_PATH

HOA_BINH_DATA_FOLDER = PRJ_PATH / "data" / "hoa_binh"


# TODO: Add description for each field here and bellow.
class SpectralComponent(Enum):
    WL = "WL"
    Lw1 = "Lw1"
    Lw2 = "Lw2"
    Lw3 = "Lw3"
    Ls1 = "Ls1"
    Ls2 = "Ls2"
    Ls3 = "Ls3"
    Lw = "Lw"
    Ls = "Ls"
    Lp = "Lp"
    Rw = "Rw"


class SpectralSample(NamedTuple):
    """
    Instance to store spectral data at a specific wave length of a sample.
    """
    wave_length: float
    lw1: float
    lw2: float
    lw3: float
    ls1: float
    ls2: float
    ls3: float
    lw: float
    ls: float
    lp: float
    rw: float


class ChemicalSample(NamedTuple):
    """
    Instance to store chemical data of a sample.
    """
    sd: float
    tss: float
    tp: float
    chla: float


class HoaBinhSample(NamedTuple):
    """
    Instance to store a water quality sample.
    """
    name: str
    # Spectral data.
    spectral_data: List[SpectralSample]
    # Chemical data.
    chemical_data: ChemicalSample


def load_2023_chemical_data(df: pd.DataFrame) -> Dict[str, ChemicalSample]:
    """
    Load chemical data from a dataframe.
    :return: Mapping between sample name and the corresponding chemical data.
    """
    chemical_data: Dict[str, ChemicalSample] = {}
    for _, row in df.iterrows():
        chemical_data[row["Sample"]] = ChemicalSample(sd=row["SD (m)"], tss=row["TSS (mg/L)"], tp=row["TP (mg*m-3)"],
                                                      chla=row["Chla (mg*m-3)"])
    return chemical_data


def load_spectral_data(spectral_df: pd.DataFrame) -> List[SpectralSample]:
    """
    Load spectral data from a dataframe.
    """
    # List of spectral sample for different wave lengths.
    spectral_data: List[SpectralSample] = []
    # Column name corresponding to "Rw".
    rw_col = [col for col in spectral_df.columns if SpectralComponent.Rw.value in col][0]
    for _, row in spectral_df.iterrows():
        spectral_data.append(
            SpectralSample(wave_length=row[SpectralComponent.WL.value],
                           lw1=row[SpectralComponent.Lw1.value],
                           lw2=row[SpectralComponent.Lw2.value] if SpectralComponent.Lw2.value in row.keys() else None,
                           lw3=row[SpectralComponent.Lw3.value] if SpectralComponent.Lw3.value in row.keys() else None,
                           ls1=row[SpectralComponent.Ls1.value],
                           ls2=row[SpectralComponent.Ls2.value] if SpectralComponent.Ls2.value in row.keys() else None,
                           ls3=row[SpectralComponent.Ls3.value] if SpectralComponent.Ls3.value in row.keys() else None,
                           ls=row[SpectralComponent.Ls.value] if SpectralComponent.Ls.value in row.keys() else None,
                           lw=row[SpectralComponent.Lw.value] if SpectralComponent.Lw.value in row.keys() else None,
                           lp=row[SpectralComponent.Lp.value],
                           rw=row[rw_col]))
    return spectral_data


# Because the data format is not the same between 2023 and 2024 so we need separate functions for loading data.
def process_data_2023() -> List[HoaBinhSample]:
    """
    Load all data of 2023.
    """
    data_2023_file = HOA_BINH_DATA_FOLDER / "data - ho hoa binh - 13&14July2023.xlsx"
    xls = pd.ExcelFile(data_2023_file)

    samples: List[HoaBinhSample] = []

    chemical_df = xls.parse("GPS + WQP")
    chemical_data = load_2023_chemical_data(chemical_df)
    for sample_name, chemical_sample in tqdm(chemical_data.items()):
        spectral_df = xls.parse(sample_name)
        spectral_data = load_spectral_data(spectral_df)
        samples.append(HoaBinhSample(name=sample_name, spectral_data=spectral_data, chemical_data=chemical_sample))
    return samples


def process_2024_result() -> Dict[str, ChemicalSample]:
    """
    Load chemical results from result sheet.
    """
    result_2024_file = HOA_BINH_DATA_FOLDER / "Ket qua phan tich mau nuoc ho Hoa Binh.xlsx"
    df = pd.read_excel(result_2024_file).iloc[5:, :]
    chemical_samples: Dict[str, ChemicalSample] = {}
    for ii in range(df.shape[0]):
        sample_name = df.iloc[ii, 0]
        chemical_samples[sample_name] = ChemicalSample(sd=df.iloc[ii, 3], chla=df.iloc[ii, 4], tp=df.iloc[ii, 5],
                                                       tss=df.iloc[ii, 6])
    return chemical_samples


def process_data_2024() -> List[HoaBinhSample]:
    """
    Process data of 2034.
    """
    chemical_samples = process_2024_result()
    data_2024_file = HOA_BINH_DATA_FOLDER / "2024 - Ho Hoa Binh - 0510 Pho.xlsx"
    xls = pd.ExcelFile(data_2024_file)
    # The first sheet is the summary sheet.
    all_sample_names = xls.sheet_names[1:]
    sheet_name_mapping = {"HB3": "HB03", "HB4": "HB04", "HB5": "HB05", "HB6": "HB06", "HB7": "HB07", "HB8": "HB08",
                          "HB9": "HB09"}

    samples: List[HoaBinhSample] = []
    for sheet_name in tqdm(all_sample_names):
        actual_sample_name = sheet_name_mapping.get(sheet_name, sheet_name)
        spectral_data = load_spectral_data(xls.parse(sheet_name))
        samples.append(HoaBinhSample(name=actual_sample_name, chemical_data=chemical_samples[actual_sample_name],
                                     spectral_data=spectral_data))

    return samples
