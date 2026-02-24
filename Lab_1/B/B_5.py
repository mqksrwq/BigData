import pandas as pd

df = pd.read_csv('2_taxi_nyc.csv', sep=',')
df.columns = df.columns.str.strip().str.replace(' ', '_')

# Группировка по borough и hday
grouped = df.groupby(['borough', 'hday'])['pickups'].mean().unstack()

# Сравнение
more_on_holiday = grouped[grouped['Y'] > grouped['N']]

print("Средние поездки по группам:\n", grouped.round(2))
print("\nРайоны с большим средним на праздниках:\n",
      more_on_holiday.index.tolist())
