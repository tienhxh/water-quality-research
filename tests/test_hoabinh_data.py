from torch import nn

from data_processing.hoa_binh_data_processing import process_data_2023, process_data_2024
from models.regression.simple_regression import BasicRegression
from utils.dataset import get_hoabinh_1D_dataloader

input_size = 500

data_2023 = process_data_2023()
train_dataloader = get_hoabinh_1D_dataloader(data_2023, spectral_field="rw", input_size=input_size)

data_2024 = process_data_2024()
test_dataloader = get_hoabinh_1D_dataloader(data_2024, spectral_field="rw", input_size=input_size)

model = BasicRegression(input_dim=input_size)

num_epochs = 200
lr = 0.01
lr_params = {"start_factor": 0.1,
             "end_factor": lr, "total_iters": num_epochs}
model.fit(train_dataloader, test_dataloader, epochs=num_epochs, optimizer_type="adam", loss_fn=nn.MSELoss(), lr=lr,
          scheduler_params=lr_params, patience=20)
