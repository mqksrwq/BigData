import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Настройка отображения датафрейма в консоли
pd.set_option('display.max_rows', None)  # Показывать все строки
pd.set_option('display.max_columns', None)  # Показывать все колонки
pd.set_option('display.width', None)  # Авто-ширина
pd.set_option('display.float_format', '{:.3f}'.format)  # Формат чисел

# 1. ОЧИСТКА ДАННЫХ
print("=" * 80)
print("ЭТАП 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
print("=" * 80)

# 1.1. Загрузка исходных данных
df_source = pd.read_csv('taxi_peru.csv', sep=';',
                        parse_dates=['start_at', 'end_at', 'arrived_at'],
                        low_memory=False)

print(f"Загружено строк: {len(df_source)}")
print(f"Уникальных водителей (всего): {df_source['driver_id'].nunique()}")

# 1.2. Выборка необходимых признаков
necessary_columns = [
    'driver_id', 'journey_id', 'end_state', 'driver_score',
    'start_lat', 'start_lon', 'end_lat', 'end_lon'
]

# Проверка целостности данных
missing_cols = set(necessary_columns) - set(df_source.columns)
if missing_cols:
    raise ValueError(f"❌ Отсутствуют колонки: {missing_cols}")

# 1.3. Создание промежуточного датафрейма
df_work = df_source[necessary_columns].copy()

# 1.4. Базовая валидация
df_work = df_work[df_work['driver_id'].notna()].copy()

print(f"Рабочий датасет: {len(df_work)} строк, {len(df_work.columns)} колонок")

# 2. РАЗВЕДЫВАТЕЛЬНЫЙ АНАЛИЗ И ПРЕДВАРИТЕЛЬНАЯ ФИЛЬТРАЦИЯ
print("\n" + "=" * 80)
print("ЭТАП 2: АГРЕГАЦИЯ И РАСЧЕТ МЕТРИК")
print("=" * 80)

# 2.1. Определение валидных состояний поездок
valid_states = ['drop off', 'not found', 'rider cancel', 'failure']

# 2.2. Агрегация статистики по водителям
driver_stats = df_work.groupby('driver_id').agg(
    trips_count=('journey_id', 'count'),
    success_trips=('end_state', lambda x: (x == 'drop off').sum()),
    not_found_trips=('end_state', lambda x: (x == 'not found').sum()),
    avg_driver_score=('driver_score', 'mean'),
    rated_count=('driver_score', lambda x: x.notna().sum())
).reset_index()

print(f"Уникальных водителей после агрегации: {len(driver_stats)}")

# 2.3. Предварительная фильтрация
driver_stats = driver_stats[driver_stats['trips_count'] >= 5].copy()
print(f"Водителей после фильтра надежности (≥5 поездок): {len(driver_stats)}")

# 2.4. Расчет метрик качества
driver_stats['completion_rate'] = driver_stats['success_trips'] / driver_stats['trips_count']
driver_stats['not_found_rate'] = driver_stats['not_found_trips'] / driver_stats['trips_count']
driver_stats['rated_share'] = driver_stats['rated_count'] / driver_stats['trips_count']

# 2.5. Нормализация показателей к диапазону [0, 1]
driver_stats['rating_norm'] = np.clip(driver_stats['avg_driver_score'] / 5.0, 0, 1)
driver_stats['completion_norm'] = np.clip(driver_stats['completion_rate'], 0, 1)
driver_stats['nofound_norm'] = np.clip(1 - driver_stats['not_found_rate'], 0, 1)

# 2.6. Расчет коэффициента доверия
N = 50  # Базовое количество поездок для полного доверия
driver_stats['trust_coef'] = np.minimum(1.0, driver_stats['trips_count'] / N)

# 2.7. Расчет взвешенного скоринга
# Веса
w1, w2, w3 = 0.6, 0.25, 0.15
driver_stats['score_0_1'] = (
        w1 * driver_stats['rating_norm'] +
        w2 * driver_stats['completion_norm'] +
        w3 * driver_stats['nofound_norm']
)

# 2.8. Финальный рейтинг с учетом доверия
driver_stats['driver_rating_0_5'] = 5.0 * driver_stats['score_0_1'] * driver_stats['trust_coef']
driver_stats['driver_rating_0_5'] = np.round(driver_stats['driver_rating_0_5'], 2)

# 2.9. Сортировка по убыванию рейтинга
driver_stats = driver_stats.sort_values('driver_rating_0_5', ascending=False).reset_index(drop=True)

print(f"Метрики рассчитаны для {len(driver_stats)} водителей")
print(
    f"Диапазон рейтингов: [{driver_stats['driver_rating_0_5'].min():.2f}, {driver_stats['driver_rating_0_5'].max():.2f}]")

# 3. ВЫДЕЛЕНИЕ УСЛОВНОЙ ВЕРХУШКИ
print("\n" + "=" * 80)
print("ЭТАП 3: СЕГМЕНТАЦИЯ ЛУЧШИХ ВОДИТЕЛЕЙ")
print("=" * 80)

# 3.1. Определение порога отсечения через квантиль (верхние 10%)
threshold_percentile = 0.90
rating_threshold = driver_stats['driver_rating_0_5'].quantile(threshold_percentile)

# 3.2. Фильтрация элиты
if len(driver_stats) > 0:
    df_elite = driver_stats[driver_stats['driver_rating_0_5'] >= max(rating_threshold, 0.1)].copy()
else:
    df_elite = pd.DataFrame(columns=driver_stats.columns)

print(f"Порог входа в элиту (квантиль {threshold_percentile * 100:.0f}%): {rating_threshold:.2f}")
print(f"Водителей в элите: {len(df_elite)}")

# 4. ИТОГОВЫЙ РЕЗУЛЬТАТ
print("\n" + "=" * 80)
print("ЭТАП 4: ФОРМИРОВАНИЕ ФИНАЛЬНОГО ОТЧЕТА")
print("=" * 80)

# 4.1. Выбор ключевых колонок для финального датафрейма
result_columns = [
    'driver_id',
    'driver_rating_0_5',
    'trips_count',
    'completion_rate',
    'not_found_rate',
    'avg_driver_score',
    'trust_coef'
]

# 4.2. Формирование итогового датафрейма
final_best_drivers_df = df_elite[result_columns].reset_index(drop=True)

# 4.3. Вывод статистики отчета
print(f"\nВСЕГО ВОДИТЕЛЕЙ В АНАЛИЗЕ: {len(driver_stats)}")
print(f"ВОДИТЕЛЕЙ В ЭЛИТЕ: {len(final_best_drivers_df)}")
print(f"СРЕДНИЙ РЕЙТИНГ ЭЛИТЫ: {final_best_drivers_df['driver_rating_0_5'].mean():.2f}")
print(f"МИН. РЕЙТИНГ В ЭЛИТЕ: {final_best_drivers_df['driver_rating_0_5'].min():.2f}")
print(f"МАКС. РЕЙТИНГ В ЭЛИТЕ: {final_best_drivers_df['driver_rating_0_5'].max():.2f}")

# 4.4. Полный вывод водителей элиты в консоль
print("\n" + "=" * 80)
print("ЭТАП 5: ПОЛНЫЙ СПИСОК ВСЕХ ВОДИТЕЛЕЙ ЭЛИТЫ")
print("=" * 80)

if len(final_best_drivers_df) > 0:
    print(final_best_drivers_df.to_string(index=True))
else:
    print("Нет водителей, удовлетворяющих критериям элиты")
