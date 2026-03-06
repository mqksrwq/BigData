import pandas as pd

df = pd.read_csv('2_taxi_nyc.csv', sep=',')
df.columns = df.columns.str.strip().str.replace(' ', '_')

summary = df.groupby('borough')[['pickups', 'sd']].agg(['mean', 'sum']).round(2)

print("=== СВОДКА ПО РАЙОНАМ ===")
print(summary)
