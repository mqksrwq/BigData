import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# Настройка отображения
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)


# ФУНКЦИЯ: ПОИСК ТОЧКИ ПЕРЕГИБА
def find_elbow_point(y_values, method='distance'):
    y = np.array(y_values)
    n = len(y)

    if n < 10:
        return n // 2, np.median(y)

    x = np.arange(n)

    if method == 'distance':
        x_start, y_start = x[0], y[0]
        x_end, y_end = x[-1], y[-1]
        line_vec = np.array([x_end - x_start, y_end - y_start])
        line_len = np.linalg.norm(line_vec)

        if line_len == 0:
            return n // 2, y[n // 2]

        line_unit = line_vec / line_len
        distances = np.zeros(n)
        for i in range(n):
            point_vec = np.array([x[i] - x_start, y[i] - y_start])
            proj = np.abs(np.cross(line_unit, point_vec))
            distances[i] = proj
        elbow_idx = np.argmax(distances)

    elif method == 'curvature':
        window = min(11, n // 5)
        if window % 2 == 0:
            window += 1
        if window >= n:
            window = n - 2 if n > 2 else 3
        y_smooth = savgol_filter(y, window, polyorder=2)
        dy = np.gradient(y_smooth)
        ddy = np.gradient(dy)
        curvature = np.abs(ddy) / (1 + dy ** 2) ** 1.5
        search_range = slice(n // 10, 9 * n // 10)
        elbow_idx = np.argmax(curvature[search_range]) + search_range.start

    elif method == 'kneedle':
        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())
        diff = y_norm - x_norm
        diff_deriv = np.gradient(diff)
        diff_dderiv = np.gradient(diff_deriv)
        start_idx = n // 10
        candidates = np.where(diff_dderiv[start_idx:] < 0)[0]
        if len(candidates) > 0:
            elbow_idx = start_idx + candidates[0]
        else:
            elbow_idx = np.argmax(diff[start_idx:]) + start_idx
    else:
        raise ValueError(f"Неизвестный метод: {method}")

    elbow_idx = np.clip(elbow_idx, 1, n - 2)
    return elbow_idx, y[elbow_idx]


# ФУНКЦИЯ: АДАПТИВНЫЙ ПОИСК ПОРОГА ЭЛИТЫ
def find_adaptive_elite_threshold(driver_stats, min_trips=100,
                                  target_min_elite=5,
                                  max_iterations=10):
    ratings = driver_stats['driver_rating_0_5'].values
    n_drivers = len(driver_stats)

    # Шаг 1: Пробуем аналитический порог
    elbow_idx, elbow_threshold = find_elbow_point(ratings, method='distance')
    current_threshold = elbow_threshold
    method_log = [f"Start: Elbow Method = {current_threshold:.3f}"]

    # Шаг 2: Проверяем, сколько водителей проходит фильтр
    def count_elite(threshold):
        mask = (driver_stats['driver_rating_0_5'] >= threshold) & \
               (driver_stats['trips_count'] >= min_trips)
        return mask.sum()

    elite_count = count_elite(current_threshold)
    method_log.append(f"Elite at elbow threshold: {elite_count} drivers")

    # Шаг 3: Если элита пуста или слишком мала — адаптируем порог
    iteration = 0
    while elite_count < target_min_elite and iteration < max_iterations:
        # Снижаем порог на 5% от текущего диапазона
        rating_range = driver_stats['driver_rating_0_5'].max() - driver_stats['driver_rating_0_5'].min()
        step = rating_range * 0.05
        current_threshold = max(current_threshold - step, driver_stats['driver_rating_0_5'].min())

        elite_count = count_elite(current_threshold)
        iteration += 1
        method_log.append(f"Iter {iteration}: threshold={current_threshold:.3f}, elite={elite_count}")

    # Шаг 4: Финальная проверка
    if elite_count == 0:
        candidates = driver_stats[driver_stats['trips_count'] >= min_trips].copy()
        if len(candidates) > 0:
            candidates = candidates.sort_values('driver_rating_0_5', ascending=False)
            final_threshold = candidates.iloc[min(target_min_elite - 1, len(candidates) - 1)]['driver_rating_0_5']
            method_log.append(f"Fallback: top-{target_min_elite} threshold = {final_threshold:.3f}")
        else:
            final_threshold = driver_stats['driver_rating_0_5'].quantile(0.90)
            method_log.append(f"Critical fallback: 90th percentile = {final_threshold:.3f}")
            min_trips = min(50, driver_stats['trips_count'].quantile(0.75))
    else:
        final_threshold = current_threshold

    # Формируем итоговую элиту
    elite_mask = (driver_stats['driver_rating_0_5'] >= final_threshold) & \
                 (driver_stats['trips_count'] >= min_trips)
    df_elite = driver_stats[elite_mask].copy().sort_values('driver_rating_0_5', ascending=False).reset_index(drop=True)

    # Описание метода
    if iteration == 0:
        method_used = "Elbow Method (без адаптации)"
    elif elite_count > 0:
        method_used = (f"Elbow + адаптация ({iteration} итераций, порог снижен на "
                       f"{elbow_threshold - final_threshold:.3f})")
    else:
        method_used = "Fallback: топ-5 по рейтингу"

    return final_threshold, df_elite, method_used, method_log


# БЛОК 1: ЗАГРУЗКА И ОЧИСТКА ДАННЫХ
print("=" * 80)
print("ЭТАП 1: ЗАГРУЗКА И ОЧИСТКА ДАННЫХ")
print("=" * 80)

df_source = pd.read_csv('taxi_peru.csv', sep=';',
                        parse_dates=['start_at', 'end_at', 'arrived_at'],
                        low_memory=False)

print(f"Загружено строк: {len(df_source):,}")
print(f"Уникальных водителей: {df_source['driver_id'].nunique()}")

print("\nОчистка:")
cleaning_log = {}

for col in ['driver_id', 'journey_id']:
    missing = df_source[col].isna().sum()
    cleaning_log[f'missing_{col}'] = missing
    df_source = df_source[df_source[col].notna()].copy()
    print(f"Удалено без {col}: {missing:,}")

missing_state = df_source['end_state'].isna().sum()
df_source = df_source[df_source['end_state'].notna()]
print(f"Удалено без end_state: {missing_state:,}")

invalid_scores = ((df_source['driver_score'] < 0) | (df_source['driver_score'] > 5)).sum()
df_source.loc[(df_source['driver_score'] < 0) | (df_source['driver_score'] > 5), 'driver_score'] = np.nan
print(f"Аномальные оценки → NaN: {invalid_scores:,}")

zero_coords = ((df_source['start_lat'] == 0) & (df_source['start_lon'] == 0)).sum()
df_source.loc[(df_source['start_lat'] == 0) & (df_source['start_lon'] == 0),
['start_lat', 'start_lon']] = np.nan
print(f"Нулевые координаты → NaN: {zero_coords:,}")

print(f"\nПосле очистки: {len(df_source):,} строк")

necessary_columns = [
    'driver_id', 'journey_id', 'end_state', 'driver_score',
    'start_lat', 'start_lon', 'end_lat', 'end_lon'
]
df_work = df_source[necessary_columns].copy()

# БЛОК 2: РАСЧЕТ МЕТРИК
print("\n" + "=" * 80)
print("ЭТАП 2: АГРЕГАЦИЯ И РАСЧЕТ МЕТРИК")
print("=" * 80)

driver_stats = df_work.groupby('driver_id').agg(
    trips_count=('journey_id', 'count'),
    success_trips=('end_state', lambda x: (x == 'drop off').sum()),
    not_found_trips=('end_state', lambda x: (x == 'not found').sum()),
    avg_driver_score=('driver_score', 'mean'),
    rated_count=('driver_score', lambda x: x.notna().sum())
).reset_index()

print(f"Водителей после агрегации: {len(driver_stats)}")

MIN_TRIPS = 5
driver_stats = driver_stats[driver_stats['trips_count'] >= MIN_TRIPS].copy()
print(f"После фильтра (≥{MIN_TRIPS} поездок): {len(driver_stats)}")

driver_stats['completion_rate'] = driver_stats['success_trips'] / driver_stats['trips_count']
driver_stats['not_found_rate'] = driver_stats['not_found_trips'] / driver_stats['trips_count']

driver_stats['rating_norm'] = np.clip(driver_stats['avg_driver_score'] / 5.0, 0, 1)
driver_stats['completion_norm'] = np.clip(driver_stats['completion_rate'], 0, 1)
driver_stats['nofound_norm'] = np.clip(1 - driver_stats['not_found_rate'], 0, 1)

N_MIN = 100
driver_stats['trust_coef'] = np.minimum(1.0, driver_stats['trips_count'] / N_MIN)

WEIGHTS = {'rating': 0.60, 'completion': 0.25, 'not_found': 0.15}
driver_stats['score_0_1'] = (
        WEIGHTS['rating'] * driver_stats['rating_norm'] +
        WEIGHTS['completion'] * driver_stats['completion_norm'] +
        WEIGHTS['not_found'] * driver_stats['nofound_norm']
)

SCALE_FACTOR = 5.0
driver_stats['driver_rating_0_5'] = SCALE_FACTOR * driver_stats['score_0_1'] * driver_stats['trust_coef']
driver_stats['driver_rating_0_5'] = np.round(driver_stats['driver_rating_0_5'], 2)

driver_stats = driver_stats.sort_values('driver_rating_0_5', ascending=False).reset_index(drop=True)

print(f"Рейтинг рассчитан для {len(driver_stats)} водителей")
print(f"Диапазон: [{driver_stats['driver_rating_0_5'].min():.2f}, {driver_stats['driver_rating_0_5'].max():.2f}]")

# БЛОК 3: АДАПТИВНЫЙ ПОИСК ЭЛИТЫ
print("\n" + "=" * 80)
print("ЭТАП 3: АДАПТИВНЫЙ ПОИСК ЭЛИТЫ")
print("=" * 80)

# Параметры элиты
ELITE_MIN_TRIPS = 300  # Минимум поездок для элиты
TARGET_MIN_ELITE = 5  # Хотя бы 5 водителей
MAX_ADAPT_ITERATIONS = 10  # Максимально итераций подстройки порога

print(f"Параметры поиска:")
print(f"Мин. поездок для элиты: {ELITE_MIN_TRIPS}")
print(f"Целевой минимум элиты: {TARGET_MIN_ELITE} водителей")
print(f"Макс. итераций адаптации: {MAX_ADAPT_ITERATIONS}")

# Запускаем адаптивный поиск
analytical_threshold, df_elite, method_used, method_log = find_adaptive_elite_threshold(
    driver_stats,
    min_trips=ELITE_MIN_TRIPS,
    target_min_elite=TARGET_MIN_ELITE,
    max_iterations=MAX_ADAPT_ITERATIONS
)

print(f"\nИтоговый порог элиты: {analytical_threshold:.3f}")
print(f"\nЛог адаптации:")
for log_entry in method_log:
    print(f"{log_entry}")

# Статистика по опыту водителей
drivers_with_100_trips = (driver_stats['trips_count'] >= ELITE_MIN_TRIPS).sum()
print(f"\nКонтекст:")
print(f"Водителей с ≥{ELITE_MIN_TRIPS} поездок: {drivers_with_100_trips}")
print(
    f"Их средний рейтинг: {driver_stats[driver_stats['trips_count'] >= ELITE_MIN_TRIPS]['driver_rating_0_5'].mean():.2f}")
print(
    f"Их медианный рейтинг: {driver_stats[driver_stats['trips_count'] >= ELITE_MIN_TRIPS]['driver_rating_0_5'].median():.2f}")

# БЛОК 4: ВИЗУАЛИЗАЦИЯ
print("\nПостроение графика...")

ratings_sorted = driver_stats['driver_rating_0_5'].values
n_drivers = len(ratings_sorted)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Анализ элиты водителей (адаптивный порог)', fontsize=16, fontweight='bold')

# График 1: Ранжированный рейтинг с адаптивным порогом
ax = axes[0, 0]
x = np.arange(1, n_drivers + 1)
ax.plot(x, ratings_sorted, linewidth=1.5, color='#2E86AB', label='Рейтинг водителя', zorder=2)

# Вертикальная линия — позиция порога в рейтинге
if len(df_elite) > 0:
    elite_min_pos = driver_stats[driver_stats['driver_id'].isin(df_elite['driver_id'])].index.min() + 1
else:
    elite_min_pos = None

ax.axhline(y=analytical_threshold, color='#E94F37', linestyle='--', linewidth=2.5,
           label=f'Адаптивный порог: {analytical_threshold:.2f}', zorder=3)
ax.axhline(y=driver_stats['driver_rating_0_5'].quantile(0.90), color='#7209B7', linestyle=':', linewidth=1,
           label=f'90-й перцентиль: {driver_stats["driver_rating_0_5"].quantile(0.90):.2f}', zorder=1)

# Заливка элиты
if len(df_elite) > 0:
    elite_positions = driver_stats[driver_stats['driver_id'].isin(df_elite['driver_id'])].index
    for pos in elite_positions:
        ax.axvspan(pos, pos + 1, alpha=0.15, color='#E94F37')

ax.set_xlabel('Позиция в рейтинге (1 = лучший)', fontsize=10)
ax.set_ylabel('Рейтинг (0–5)', fontsize=10)
ax.set_title('Ранжированное распределение рейтинга', fontsize=11, fontweight='bold')
ax.legend(fontsize=9, frameon=True)
ax.grid(alpha=0.3)
ax.set_xlim(1, n_drivers)

# График 2: Рейтинг против поездок с выделением элиты
ax = axes[0, 1]
ax.scatter(driver_stats['trips_count'], driver_stats['driver_rating_0_5'],
           c=driver_stats['driver_rating_0_5'], cmap='viridis', s=30, alpha=0.5, edgecolors='white', linewidth=0.3)

# Выделяем элиту
if len(df_elite) > 0:
    ax.scatter(df_elite['trips_count'], df_elite['driver_rating_0_5'],
               c='#E94F37', s=100, edgecolors='black', linewidth=2,
               label=f'Элита ({len(df_elite)} водителей)', zorder=5)

ax.axhline(y=analytical_threshold, color='#E94F37', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(x=ELITE_MIN_TRIPS, color='#F18F01', linestyle=':', linewidth=1.5, alpha=0.7,
           label=f'Мин. поездок: {ELITE_MIN_TRIPS}')
ax.set_xlabel('Количество поездок (лог. шкала)', fontsize=10)
ax.set_ylabel('Рейтинг', fontsize=10)
ax.set_title('Рейтинг vs Опыт', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xscale('log')

# График 3: Гистограмма рейтингов водителей с ≥100 поездок
ax = axes[1, 0]
candidates = driver_stats[driver_stats['trips_count'] >= ELITE_MIN_TRIPS]['driver_rating_0_5']
if len(candidates) > 0:
    bins = min(20, len(candidates) // 3)
    ax.hist(candidates, bins=bins, edgecolor='white', color='#7209B7', alpha=0.7, label=f'≥{ELITE_MIN_TRIPS} поездок')
    ax.axvline(x=analytical_threshold, color='#E94F37', linestyle='--', linewidth=2.5,
               label=f'Порог элиты: {analytical_threshold:.2f}')
    ax.set_xlabel('Рейтинг', fontsize=10)
    ax.set_ylabel('Количество водителей', fontsize=10)
    ax.set_title(f'Распределение рейтингов (водители с ≥{ELITE_MIN_TRIPS} поездок)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
else:
    ax.text(0.5, 0.5, f'Нет водителей\nс ≥{ELITE_MIN_TRIPS} поездок',
            transform=ax.transAxes, ha='center', va='center', fontsize=12)
    ax.set_title(f'Распределение (≥{ELITE_MIN_TRIPS} поездок)', fontsize=11, fontweight='bold')

# График 4: Детали элиты
ax = axes[1, 1]
if len(df_elite) > 0:
    # Столбчатая диаграмма рейтингов элиты
    y_pos = np.arange(len(df_elite))
    bars = ax.barh(y_pos, df_elite['driver_rating_0_5'].values,
                   color=plt.cm.viridis(df_elite['trust_coef'].values), edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"#{i + 1}" for i in range(len(df_elite))], fontsize=8)
    ax.set_xlabel('Рейтинг', fontsize=10)
    ax.set_ylabel('Позиция в элите', fontsize=10)
    ax.set_title('Рейтинг водителей элиты', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

    # Подписи с количеством поездок
    for i, (idx, row) in enumerate(df_elite.iterrows()):
        ax.text(row['driver_rating_0_5'] + 0.05, i, f"{int(row['trips_count'])} поездок",
                va='center', fontsize=8)
else:
    ax.text(0.5, 0.5, 'Элита пуста', transform=ax.transAxes, ha='center', va='center', fontsize=14)
    ax.set_title('Детали элиты', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('driver_elite_adaptive_analysis.png', dpi=150, bbox_inches='tight')
print("График сохранен: 'driver_elite_adaptive_analysis.png'")
plt.show()

# БЛОК 5: ФИНАЛЬНЫЙ ОТЧЕТ И ВЫВОД ДАТАФРЕЙМА
print("\n" + "=" * 80)
print("ЭТАП 4: ФИНАЛЬНЫЙ ОТЧЕТ")
print("=" * 80)

print(f"\nИтоговые критерии элиты:")
print(f"Рейтинг ≥ {analytical_threshold:.3f} ({method_used})")
print(f"Поездок ≥ {ELITE_MIN_TRIPS}")

print(f"\nРезультаты отбора:")
print(f"Всего водителей в анализе: {len(driver_stats)}")
print(f"Водителей с ≥{ELITE_MIN_TRIPS} поездок: {drivers_with_100_trips}")
print(f"Водителей в элите: {len(df_elite)}")

if len(df_elite) > 0:
    pct = 100 * len(df_elite) / len(driver_stats)
    print(f"Доля элиты: {pct:.2f}% от всех водителей")

    print(f"\nСтатистика элиты:")
    print(
        f"Средний рейтинг: {df_elite['driver_rating_0_5'].mean():.2f} ± {df_elite['driver_rating_0_5'].std():.2f}")
    print(f"Медиана рейтинга: {df_elite['driver_rating_0_5'].median():.2f}")
    print(f"Диапазон: [{df_elite['driver_rating_0_5'].min():.2f}, {df_elite['driver_rating_0_5'].max():.2f}]")
    print(
        f"Среднее число поездок: {df_elite['trips_count'].mean():.0f} (медиана: {df_elite['trips_count'].median():.0f})")
    print(f"Средняя завершённость: {df_elite['completion_rate'].mean() * 100:.1f}%")
    print(f"Средняя оценка: {df_elite['avg_driver_score'].mean():.2f}/5.0")
    print(f"Средний trust_coef: {df_elite['trust_coef'].mean():.3f}")

# ВЫВОД ПОЛНОГО ДАТАФРЕЙМА С ВОДИТЕЛЯМИ ЭЛИТЫ
result_columns = [
    'driver_id', 'driver_rating_0_5', 'trips_count',
    'completion_rate', 'not_found_rate', 'avg_driver_score', 'trust_coef'
]
final_best_drivers_df = df_elite[result_columns].copy().reset_index(drop=True)

print("\n" + "=" * 80)
print("ПОЛНЫЙ ДАТАФРЕЙМ: ВОДИТЕЛИ ЭЛИТЫ")
print("=" * 80)

if len(final_best_drivers_df) > 0:
    pd.set_option('display.float_format', '{:.3f}'.format)

    print(f"\nВсего записей: {len(final_best_drivers_df)}\n")
    print(final_best_drivers_df.to_string(index=True))

    pd.set_option('display.float_format', '{:.4f}'.format)
else:
    print("\nЭлита пуста — ни один водитель не прошёл фильтры")
