import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


# Функция расчета рейтинга водителей
def calculate_driver_rating(df):
    # Фильтр строк
    df_valid = df[df['driver_id'].notna()].copy()

    # Валидные состояния
    valid_states = ['drop off', 'not found', 'rider cancel', 'failure']
    df_valid['valid_state'] = df_valid['end_state'].isin(valid_states)

    # Группировка
    driver_stats = df_valid.groupby('driver_id').agg(
        trips_count=('journey_id', 'count'),
        success_trips=('end_state', lambda x: (x == 'drop off').sum()),
        not_found_trips=('end_state', lambda x: (x == 'not found').sum()),
        avg_driver_score=('driver_score', 'mean'),
        rated_count=('driver_score', lambda x: x.notna().sum()),
        start_lat=('start_lat', 'first'),
        start_lon=('start_lon', 'first'),
        end_lat=('end_lat', 'first'),
        end_lon=('end_lon', 'first')
    ).reset_index()

    # Фильтр по валидным заказам
    driver_stats = driver_stats[driver_stats['trips_count'] >= 5]

    # Метрики
    N = 50

    driver_stats['completion_rate'] = driver_stats['success_trips'] / driver_stats['trips_count']
    driver_stats['not_found_rate'] = driver_stats['not_found_trips'] / driver_stats['trips_count']
    driver_stats['rated_share'] = driver_stats['rated_count'] / driver_stats['trips_count']

    # Нормализация в [0,1]
    driver_stats['rating_norm'] = np.clip(driver_stats['avg_driver_score'] / 5.0, 0, 1)
    driver_stats['completion_norm'] = np.clip(driver_stats['completion_rate'], 0, 1)
    driver_stats['nofound_norm'] = np.clip(1 - driver_stats['not_found_rate'], 0, 1)

    # Коэффициент доверия
    driver_stats['trust_coef'] = np.minimum(1.0, driver_stats['trips_count'] / N)

    # Условные веса характеристик
    w1, w2, w3 = 0.6, 0.25, 0.15
    driver_stats['score_0_1'] = (
            w1 * driver_stats['rating_norm'] +
            w2 * driver_stats['completion_norm'] +
            w3 * driver_stats['nofound_norm']
    )

    # Итоговый рейтинг
    driver_stats['driver_rating_0_5'] = 5.0 * driver_stats['score_0_1'] * driver_stats['trust_coef']

    # Округление
    driver_stats['driver_rating_0_5'] = np.round(driver_stats['driver_rating_0_5'], 2)

    return driver_stats.sort_values('driver_rating_0_5', ascending=False)


# === ОСНОВНОЙ КОД ===
if __name__ == "__main__":

    # Чтение csv
    print("Загружаем taxi_peru.csv...")
    df = pd.read_csv('taxi_peru.csv', sep=';',
                     parse_dates=['start_at', 'end_at', 'arrived_at'],
                     low_memory=False)

    print(f"Загружено {len(df)} строк")
    print(f"Уникальных водителей: {df['driver_id'].nunique()}")

    # Подсчет рейтингов
    ratings = calculate_driver_rating(df)

    print("\n=== ТОП-10 ЛУЧШИХ ВОДИТЕЛЕЙ ===")
    top_10 = ratings.head(10)
    print(top_10[['driver_rating_0_5', 'trips_count', 'completion_rate',
                  'avg_driver_score', 'not_found_rate']].round(3))

    print("\n=== СТАТИСТИКА ПО ВСЕМ ВОДИТЕЛЯМ ===")
    print(f"Водителей с рейтингом: {len(ratings)}")
    print(f"Средний рейтинг: {ratings['driver_rating_0_5'].mean():.2f}")
    print(f"Максимальный рейтинг: {ratings['driver_rating_0_5'].max():.2f}")
    print(f"Минимальный рейтинг: {ratings['driver_rating_0_5'].min():.2f}")

    # Топ-1 водитель
    if len(ratings) > 0:
        top_driver = ratings.iloc[0]['driver_id']
        print(f"\nТоп-1 водитель {top_driver}: рейтинг {ratings.iloc[0]['driver_rating_0_5']}")
