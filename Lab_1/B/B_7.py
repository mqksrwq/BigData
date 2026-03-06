import pandas as pd


def temp_to_celc(temp):
    return (temp - 32.0) * 5.0 / 9.0


# Пример использования на датасете
df = pd.read_csv('2_taxi_nyc.csv', sep=',')
df.columns = df.columns.str.strip().str.replace(' ', '_')

df['temp_c'] = temp_to_celc(df['temp'])

print("Первые 5 temp (F):", df['temp'].head().tolist())
print("Первые 5 temp_c (C):", df['temp_c'].round(2).head().tolist())

# Статистика
print("\nТемпература F: мин=%.1f, макс=%.1f" % (df['temp'].min(), df['temp'].max()))
print("Температура C: мин=%.1f, макс=%.1f" % (df['temp_c'].min(), df['temp_c'].max()))
