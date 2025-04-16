from torch import nn

from utils.new_data_utils import load_data, split_train_test
from utils.dataset import get_new_simple_dataloader, get_new_lstm_dataloader
from models.regression.simple_lstm import BasicLSTM

data = load_data()
training_data, testing_data = split_train_test(data, start_test_year=2004)

hidden_dim = 100
num_layers = 1
output_dim = 1
seq_length = 3
num_epochs = 500
lr = 0.01
lr_params = {"start_factor": 0.1,
             "end_factor": lr, "total_iters": num_epochs}

train_dataloader = get_new_lstm_dataloader(training_data, shuffle=True, batch_size=4, seq_length=seq_length)
test_dataloader = get_new_lstm_dataloader(testing_data, shuffle=False, batch_size=4, seq_length=seq_length)
first_x, first_y = train_dataloader.dataset[0]
input_dim = first_x.shape[-1]

model = BasicLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim, seq_length=seq_length)
model.fit(train_dataloader, test_dataloader, epochs=num_epochs, optimizer_type="adam", loss_fn=nn.MSELoss(), lr=lr,
          scheduler_params=lr_params, patience=20)

