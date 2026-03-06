import pandas as pd

# Загрузка и очистка данных
df = pd.read_csv('2_taxi_nyc.csv', sep=',')
df.columns = df.columns.str.strip().str.replace(' ', '_')

# Б6.1
pickups_by_mon_bor = (
    df.groupby(['pickup_month', 'borough'], as_index=False)['pickups']
    .sum()
    .sort_values('pickups', ascending=False)
)
print("pickups_by_mon_bor:\n", pickups_by_mon_bor.head(10))

# Б6.2
df_filtered = df.query("borough not in ['Manhattan', 'Bronx']")
pickups_filtered = (
    df_filtered.groupby(['pickup_month', 'borough'], as_index=False)['pickups']
    .sum()
    .sort_values('pickups', ascending=False)
)
print("\nБез Manhattan и Bronx:\n", pickups_filtered.head(10))
