import pandas as pd

df = pd.read_csv('2_taxi_nyc.csv', sep=',')
df.columns = df.columns.str.replace(' ', '_')

# 1. Сколько всего поездок
total_pickups = df['pickups'].sum()
print("Всего поездок:", total_pickups)

# 2. Поездки по районам
pickups_by_borough = (
    df.groupby('borough')['pickups']
    .sum()
    .sort_values(ascending=False)
)
print("\nПоездки по районам (сумма):")
print(pickups_by_borough)

# 3. В каком районе больше всего поездок
max_pickups = pickups_by_borough.idxmax()
print("\nРайон с max числом поездок:", max_pickups)

# 4. Queens vs Staten Island
queens_pickups = pickups_by_borough.get('Queens', 0)
staten_pickups = pickups_by_borough.get('Staten Island', 0)
print(f"\nQueens: {queens_pickups}, Staten Island: {staten_pickups}")
if queens_pickups > staten_pickups:
    print("Такси более востребованы в Queens")
elif staten_pickups > queens_pickups:
    print("Такси более востребованы в Staten Island")
else:
    print("В Queens и Staten Island суммарно одинаковое число поездок")
