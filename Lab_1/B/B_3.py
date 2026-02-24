import pandas as pd

df = pd.read_csv('2_taxi_nyc.csv')
df.columns = df.columns.str.replace(' ', '_')

# Частота
counts = df['borough'].value_counts()
print(counts)

# Сравнение
brooklyn = counts.get('Brooklyn', 0)
bronx = counts.get('Bronx', 0)
print(f"Brooklyn: {brooklyn}, Bronx: {bronx}")
print("Brooklyn чаще Bronx" if brooklyn > bronx else "...")

# Проценты
percent = df['borough'].value_counts(normalize=True) * 100
print(percent.round(2))
