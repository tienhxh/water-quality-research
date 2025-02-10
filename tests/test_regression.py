from utils.data_utils import load_and_split_data
from utils.dataset import get_simple_dataloader
from models.regression.simple_transformer import BasicTransformer  
from torch import nn

model_dim = 64  
num_heads = 4   
num_layers = 2  
output_dim = 1

files = ["predata.xls", "Data.xlsx"]
training_data, testing_data = load_and_split_data(files)

# Sử dụng DataLoader
train_dataloader = get_simple_dataloader(training_data, shuffle=False)
test_dataloader = get_simple_dataloader(testing_data, shuffle=False)

first_x, first_y = train_dataloader.dataset[0]
input_dim = first_x.shape[-1] 

# Sử dụng mô hình Transformer
model = BasicTransformer(input_dim=input_dim, model_dim=model_dim, num_heads=num_heads,
                         num_layers=num_layers, output_dim=output_dim)

num_epochs = 200
lr = 0.01
lr_params = {"start_factor": 0.1, "end_factor": lr, "total_iters": num_epochs}

model.fit(train_dataloader, test_dataloader, epochs=num_epochs, optimizer_type="adam",
          loss_fn=nn.MSELoss(), lr=lr, scheduler_params=lr_params, patience=20)