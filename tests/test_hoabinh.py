from data_processing.hoa_binh_data_processing import process_data_2024, process_data_2023

data_2023 = process_data_2023()
data_2024 = process_data_2024()

for sample in data_2023:
    print([spectral.wave_length for spectral in sample.spectral_data][:10])

print("--------------")
print("--------------")
for sample in data_2024:
    print([spectral.wave_length for spectral in sample.spectral_data][:10])

print(len(data_2023))
print(len(data_2024))