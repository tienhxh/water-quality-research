from utils.data_utils import load_and_split_data
from utils.lstm_dataset import get_lstm_dataloader
from models.regression.simple_anfis import BasicANFIS  # Giả sử mô hình ANFIS đã được tạo trong models/anfis.py
from torch import nn
import torch.optim as optim

# Thông số huấn luyện
num_epochs = 200
lr = 0.01
seq_length = 4

# Tải dữ liệu
files = ["predata.xls", "Data.xlsx"]
training_data, testing_data = load_and_split_data(files)

# Sử dụng DataLoader
train_dataloader = get_lstm_dataloader(training_data, seq_length=seq_length, shuffle=False)
test_dataloader = get_lstm_dataloader(testing_data, seq_length=seq_length, shuffle=False)

# Lấy kích thước đầu vào từ dữ liệu
first_x, first_y = train_dataloader.dataset[0]
input_dim = first_x.shape[-1] 

# Khởi tạo mô hình ANFIS
model = BasicANFIS(input_dim=input_dim, num_rules=5, output_dim=1)  # num_rules có thể điều chỉnh

num_epochs = 200
lr = 0.01
lr_params = {"start_factor": 0.1, "end_factor": lr, "total_iters": num_epochs}

model.fit(train_dataloader, test_dataloader, epochs=num_epochs, optimizer_type="adam",
          loss_fn=nn.MSELoss(), lr=lr, scheduler_params=lr_params, patience=20)