import pandas as pd

df = pd.read_csv('2_taxi_nyc.csv', sep=',')
df.columns = df.columns.str.strip().str.replace(' ', '_')

districts = ['Bronx', 'Manhattan', 'Brooklyn']

print("=== ТОП-2 МЕСЯЦА ПО ПОЕЗДКАМ ДЛЯ КАЖДОГО РАЙОНА ===")
for district in districts:
    top_months = (
        df[df['borough'] == district]
        .groupby('pickup_month')['pickups']
        .sum()
        .sort_values(ascending=False)
        .head(2)
    )
    print(f"\n{district}:")
    print(top_months)
    print(f"Максимум: {top_months.index[0]} ({top_months.iloc[0]:,} поездок)")
