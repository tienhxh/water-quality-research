from utils.data_utils import load_and_split_data
from utils.sarimax_dataset import dataset_to_dataframe
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load dữ liệu
files = ["predata.xls", "Data.xlsx"]
training_data, testing_data = load_and_split_data(files)

# Chuyển dữ liệu cho SARIMAX
df_train = dataset_to_dataframe(training_data)
df_test = dataset_to_dataframe(testing_data)

exog_train = df_train.drop(columns=["target_value"])
target_train = df_train["target_value"]
exog_test = df_test.drop(columns=["target_value"])
target_test = df_test["target_value"]

# Huấn luyện SARIMAX
sarimax_model = SARIMAX(target_train, exog=exog_train, order=(2, 1, 1), seasonal_order=(1, 1, 1, 12))
sarimax_result = sarimax_model.fit()

# Dự đoán trên tập huấn luyện (train loss)
train_forecast = sarimax_result.fittedvalues
train_mse = mean_squared_error(target_train, train_forecast)
print(f"Train Loss (MSE) = {train_mse:.6f}")

# Dự đoán trên tập test (val loss)
forecast = sarimax_result.forecast(steps=len(df_test), exog=exog_test)
val_mse = mean_squared_error(target_test, forecast)
print(f"Validation Loss (MSE) = {val_mse:.6f}")
