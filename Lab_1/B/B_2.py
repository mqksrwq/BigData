import pandas as pd

df = pd.read_csv('2_taxi_nyc.csv')

# Типы колонок
print(df.dtypes)

# Преобладающий тип
print("Object:", (df.dtypes == 'object').sum())
print("Numeric:", (df.dtypes != 'object').sum())
print("Преобладают numeric типы (11 из 14)")

# Колонки с пробелами
space_cols = [col for col in df.columns if ' ' in col]
print("Колонки с пробелами:", space_cols)

# Переименование
df.columns = df.columns.str.replace(' ', '_')
print("Новые колонки:", list(df.columns))

# Первые 15 строк без цикла
first15 = df.head(15)
print(first15)
