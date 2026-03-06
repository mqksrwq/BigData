import pandas as pd

df = pd.read_csv('2_taxi_nyc.csv', sep=',')
df.columns = df.columns.str.strip().str.replace(' ', '_')

bronx_pickups = (
    df[df['borough'] == 'Bronx']
    .groupby('pickup_month')['pickups']
    .sum()
    .sort_values(ascending=False)
)

print("=== ПОЕЗДКИ В BRONX ПО МЕСЯЦАМ (сумма pickups) ===")
print(bronx_pickups)
print(f"\nМАКСИМУМ: {bronx_pickups.index[0]} = {bronx_pickups.iloc[0]:,} поездок")
