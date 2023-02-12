import mlflow
logged_model = 'runs:/5ee652f52d8840c882dda4b9da07896d/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

df = pd.read_csv('data/processed/alura_real_estate.csv')
df = df.drop('price', axis=1)
predicted = loaded_model.predict(df)

df['predicted'] = predicted
df.to_csv('predicted.csv')