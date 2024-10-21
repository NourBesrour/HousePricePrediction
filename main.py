import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xg
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

object_columns_train = train_data.select_dtypes(include=['object']).columns.tolist()
object_columns_test = test_data.select_dtypes(include=['object']).columns.tolist()

label_encoder = LabelEncoder()
for column in object_columns_train:
    train_data[column] = label_encoder.fit_transform(train_data[column])

for column in object_columns_test:
    test_data[column] = label_encoder.fit_transform(test_data[column])

train_data_cl = train_data.query("GarageYrBlt.notnull()")

ss = StandardScaler()
x, y = train_data_cl.drop(columns='SalePrice', axis=1), train_data_cl['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=100)

x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

xgb_model = xg.XGBRegressor(learning_rate=0.2, max_depth=3, min_child_weight=2, n_estimators=200)
xgb_model.fit(x_train, y_train)


y_pred_train = xgb_model.predict(x_train)
y_pred_test = xgb_model.predict(x_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("XGBoost RMSE (Train):", round(rmse_train, 2))
print("XGBoost RMSE (Test):", round(rmse_test, 2))

y_pred_final_test = xgb_model.predict(ss.transform(test_data.values))

test_data['ID'] = range(1461, 1461 + len(test_data))

results_test = pd.DataFrame({
    'ID': test_data['ID'],
    'SalePrice': y_pred_final_test
})

results_test.to_csv('result_predictions.csv', index=False)
print("Result predictions saved to 'result_predictions.csv'")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_train, y=y_pred_train, color='blue', label='Train')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', linewidth=2)
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Actual vs Predicted SalePrice (Training Data)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_test, color='green', label='Test')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Actual vs Predicted SalePrice (Test Data)')
plt.legend()
plt.show()
