from torch import nn

from utils.data_utils import load_and_split_data
from utils.dataset import get_simple_dataloader
from models.regression.simple_regression import BasicRegression

hidden_dim = 100
num_layers = 1
output_dim = 1
seq_length = 4
num_rules = 4

files = ["predata.xls", "Data.xlsx"]
training_data, testing_data = load_and_split_data(files)
train_dataloader = get_simple_dataloader(training_data, shuffle=True, batch_size=16)
test_dataloader = get_simple_dataloader(testing_data, shuffle=False, batch_size=16)

first_x, first_y = train_dataloader.dataset[0]
input_dim = first_x.shape[-1]

model = BasicRegression(input_dim=input_dim)

num_epochs = 500
lr = 0.01
lr_params = {"start_factor": 0.1,
             "end_factor": lr, "total_iters": num_epochs}
model.fit(train_dataloader, test_dataloader, epochs=num_epochs, optimizer_type="adam", loss_fn=nn.MSELoss(), lr=lr,
          scheduler_params=lr_params, patience=20)
