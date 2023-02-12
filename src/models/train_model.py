import math
import mlflow
import xgboost
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/processed/alura_real_estate.csv')
x = df.drop('price', axis=1)
y = df['price']



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
dtrain = xgboost.DMatrix(X_train,label=y_train)
dtest = xgboost.DMatrix(X_test,label=y_test)

xgb_params = {
    "learning_rate": 0.2,
    "seed": 42
}

mlflow.set_experiment('house-prices-script')
with mlflow.start_run():
    mlflow.xgboost.autolog()
    xgb = xgboost.train(xgb_params, dtrain, evals=[(dtrain,"train")])
    xgb_predicted = xgb.predict(dtest)
    mse = mean_squared_error(y_test, xgb_predicted)
    r2 = r2_score(y_test, xgb_predicted)
    rmse = math.sqrt(mse)
    print(f' rmse: {rmse}, mse: {mse}, r2 {r2}')
    mlflow.log_metric("mse",mse)
    mlflow.log_metric("rmse",rmse)
    mlflow.log_metric("r2",r2)