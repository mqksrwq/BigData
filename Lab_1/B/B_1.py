import pandas as pd

# Загрузка с правильным разделителем (,)
df = pd.read_csv('2_taxi_nyc.csv')
print("Содержимое датафрейма:")
print(df)
print(f"Размерность датафрейма: {df.shape}")

# Загрузка с неправильным разделителем (;)
df_wrong = pd.read_csv('2_taxi_nyc.csv', sep=';')
print("Содержимое неправильного датафрейма:")
print(df_wrong)
print(f"Размерность неправильного: {df_wrong.shape}")
