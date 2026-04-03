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
def find_adaptive_elite_threshold(stats_df, min_rides=100,
                                  target_min_elite=5,
                                  max_iterations=10,
                                  rating_col='passenger_rating_0_5',
                                  rides_col='rides_count'):
    ratings = stats_df[rating_col].values
    n_items = len(stats_df)

    # Шаг 1: Аналитический порог
    elbow_idx, elbow_threshold = find_elbow_point(ratings, method='distance')
    current_threshold = elbow_threshold
    method_log = [f"Start: Elbow Method = {current_threshold:.3f}"]

    # Шаг 2: Подсчёт элиты
    def count_elite(threshold):
        mask = (stats_df[rating_col] >= threshold) & \
               (stats_df[rides_col] >= min_rides)
        return mask.sum()

    elite_count = count_elite(current_threshold)
    method_log.append(f"Elite at elbow threshold: {elite_count} items")

    # Шаг 3: Адаптация
    iteration = 0
    while elite_count < target_min_elite and iteration < max_iterations:
        rating_range = stats_df[rating_col].max() - stats_df[rating_col].min()
        step = rating_range * 0.05
        current_threshold = max(current_threshold - step, stats_df[rating_col].min())

        elite_count = count_elite(current_threshold)
        iteration += 1
        method_log.append(f"Iter {iteration}: threshold={current_threshold:.3f}, elite={elite_count}")

    # Шаг 4: Фоллбэк
    if elite_count == 0:
        candidates = stats_df[stats_df[rides_col] >= min_rides].copy()
        if len(candidates) > 0:
            candidates = candidates.sort_values(rating_col, ascending=False)
            final_threshold = candidates.iloc[min(target_min_elite - 1, len(candidates) - 1)][rating_col]
            method_log.append(f"Fallback: top-{target_min_elite} threshold = {final_threshold:.3f}")
        else:
            final_threshold = stats_df[rating_col].quantile(0.90)
            method_log.append(f"Critical fallback: 90th percentile = {final_threshold:.3f}")
            min_rides = min(50, stats_df[rides_col].quantile(0.75))
    else:
        final_threshold = current_threshold

    # Финальная элита
    elite_mask = (stats_df[rating_col] >= final_threshold) & \
                 (stats_df[rides_col] >= min_rides)
    df_elite = stats_df[elite_mask].copy().sort_values(rating_col, ascending=False).reset_index(drop=True)

    # Описание метода
    if iteration == 0:
        method_used = "Elbow Method (без адаптации)"
    elif elite_count > 0:
        method_used = f"Elbow + адаптация ({iteration} итер., порог снижен на {elbow_threshold - final_threshold:.3f})"
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

if 'passenger_id' in df_source.columns:
    print(f"Уникальных пассажиров: {df_source['passenger_id'].nunique()}")
else:
    print("Колонка 'passenger_id' не найдена в файле — будем генерировать тестовые данные")

print("\nОчистка данных:")
cleaning_log = {}

# Обработка passenger_id
if 'passenger_id' not in df_source.columns:
    print("Создаём симулированные passenger_id...")
    np.random.seed(42)

    # Генерация ID
    n_passengers = max(100, len(df_source) // 10)
    df_source['passenger_id'] = ['P_' + str(i).zfill(4) for i in
                                 np.random.randint(1, n_passengers + 1, size=len(df_source))]
    print(f"Сгенерировано {df_source['passenger_id'].nunique()} уникальных пассажиров")

missing_passenger = df_source['passenger_id'].isna().sum()
cleaning_log['missing_passenger_id'] = missing_passenger
df_clean = df_source[df_source['passenger_id'].notna()].copy()
print(f"Удалено записей без passenger_id: {missing_passenger:,}")

# Обработка journey_id
missing_journey = df_clean['journey_id'].isna().sum()
cleaning_log['missing_journey_id'] = missing_journey
df_clean = df_clean[df_clean['journey_id'].notna()].copy()
print(f"Удалено записей без journey_id: {missing_journey:,}")

# Обработка end_state
missing_state = df_clean['end_state'].isna().sum()
cleaning_log['missing_end_state'] = missing_state
df_clean = df_clean[df_clean['end_state'].notna()]
print(f"  • Удалено записей без end_state: {missing_state:,}")

# Обработка passenger_rating
if 'passenger_rating' not in df_clean.columns:
    print("Колонка 'passenger_rating' не найдена — генерируем симулированные оценки...")

    df_clean['passenger_rating'] = np.random.choice(
        [1, 2, 3, 4, 5], size=len(df_clean), p=[0.02, 0.03, 0.10, 0.35, 0.50]
    )

invalid_ratings = ((df_clean['passenger_rating'] < 0) | (df_clean['passenger_rating'] > 5)).sum()
cleaning_log['invalid_passenger_rating'] = invalid_ratings
df_clean.loc[(df_clean['passenger_rating'] < 0) | (df_clean['passenger_rating'] > 5), 'passenger_rating'] = np.nan
print(f"Аномальные оценки пассажиров → NaN: {invalid_ratings:,}")

print(f"\nПосле очистки: {len(df_clean):,} строк ({100 * len(df_clean) / len(df_source):.1f}% от исходных)")

passenger_columns = [
    'passenger_id', 'journey_id', 'end_state', 'passenger_rating',
    'start_lat', 'start_lon', 'end_lat', 'end_lon'
]
if 'payment_status' in df_clean.columns:
    passenger_columns.append('payment_status')

df_work = df_clean[[col for col in passenger_columns if col in df_clean.columns]].copy()
print(f"Рабочий датасет: {len(df_work.columns)} колонок")

# БЛОК 2: РАСЧЕТ МЕТРИК ДЛЯ ПАССАЖИРОВ
print("\n" + "=" * 80)
print("ЭТАП 2: АГРЕГАЦИЯ И РАСЧЕТ МЕТРИК")
print("=" * 80)

passenger_stats = df_work.groupby('passenger_id').agg(
    rides_count=('journey_id', 'count'),
    completed_rides=('end_state', lambda x: (x == 'drop off').sum()),
    canceled_rides=('end_state', lambda x: (x == 'rider cancel').sum()),
    no_show_rides=('end_state', lambda x: (x == 'no show').sum()),
    avg_passenger_rating=('passenger_rating', 'mean'),
    rated_count=('passenger_rating', lambda x: x.notna().sum())
).reset_index()

print(f"Пассажиров после агрегации: {len(passenger_stats)}")

# Фильтр надежности
MIN_RIDES = 5
passenger_stats = passenger_stats[passenger_stats['rides_count'] >= MIN_RIDES].copy()
print(f"После фильтра (≥{MIN_RIDES} поездок): {len(passenger_stats)} пассажиров")

# Базовые метрики
passenger_stats['cancellation_rate'] = passenger_stats['canceled_rides'] / passenger_stats['rides_count']
passenger_stats['no_show_rate'] = passenger_stats['no_show_rides'] / passenger_stats['rides_count']
passenger_stats['payment_issue_rate'] = passenger_stats.get('payment_issues', 0) / passenger_stats[
    'rides_count'] if 'payment_issues' in passenger_stats.columns else 0.0

# Нормализация [0,1]
passenger_stats['rating_norm'] = np.clip(passenger_stats['avg_passenger_rating'] / 5.0, 0, 1)
passenger_stats['cancel_norm'] = np.clip(1 - passenger_stats['cancellation_rate'], 0, 1)
passenger_stats['payment_norm'] = np.clip(1 - passenger_stats['payment_issue_rate'], 0, 1)

# Коэффициент доверия
N_MIN = 50
passenger_stats['trust_coef'] = np.minimum(1.0, passenger_stats['rides_count'] / N_MIN)

# Взвешенный скоринг
WEIGHTS_PASSENGER = {'rating': 0.40, 'cancellation': 0.40, 'payment': 0.20}
passenger_stats['score_0_1'] = (
        WEIGHTS_PASSENGER['rating'] * passenger_stats['rating_norm'] +
        WEIGHTS_PASSENGER['cancellation'] * passenger_stats['cancel_norm'] +
        WEIGHTS_PASSENGER['payment'] * passenger_stats['payment_norm']
)

# Итоговый рейтинг [0, 5]
SCALE_FACTOR = 5.0
passenger_stats['passenger_rating_0_5'] = SCALE_FACTOR * passenger_stats['score_0_1'] * passenger_stats['trust_coef']
passenger_stats['passenger_rating_0_5'] = np.round(passenger_stats['passenger_rating_0_5'], 2)

# Сортировка
passenger_stats = passenger_stats.sort_values('passenger_rating_0_5', ascending=False).reset_index(drop=True)

print(f"Рейтинг рассчитан для {len(passenger_stats)} пассажиров")
print(
    f"Диапазон: [{passenger_stats['passenger_rating_0_5'].min():.2f}, {passenger_stats['passenger_rating_0_5'].max():.2f}]")

# БЛОК 3: АДАПТИВНЫЙ ПОИСК ЭЛИТЫ
print("\n" + "=" * 80)
print("ЭТАП 3: АДАПТИВНЫЙ ПОИСК ЭЛИТЫ")
print("=" * 80)

ELITE_MIN_RIDES = 15

TARGET_ELITE_SIZE = 15
MAX_ELITE_PERCENT = 0.01

TARGET_MIN_ELITE = 5
MAX_ADAPT_ITERATIONS = 10

print(f"Параметры поиска:")
print(f"Мин. поездок для элиты: {ELITE_MIN_RIDES}")
print(f"Целевой размер элиты: ~{TARGET_ELITE_SIZE} пассажиров")
print(f"Макс. доля элиты: {MAX_ELITE_PERCENT * 100:.1f}% от всех")
print(f"Мин. пассажиров в элите: {TARGET_MIN_ELITE}")

# АДАПТИВНЫЙ ПОИСК С ДОПОЛНИТЕЛЬНЫМ КОНТРОЛЕМ РАЗМЕРА
analytical_threshold, df_elite, method_used, method_log = find_adaptive_elite_threshold(
    passenger_stats,
    min_rides=ELITE_MIN_RIDES,
    target_min_elite=TARGET_MIN_ELITE,
    max_iterations=MAX_ADAPT_ITERATIONS,
    rating_col='passenger_rating_0_5',
    rides_col='rides_count'
)

# ДОПОЛНИТЕЛЬНЫЙ ФИЛЬТР: ограничиваем размер элиты
if len(df_elite) > 0:
    # Вычисляем допустимый максимум по проценту
    max_elite_count = max(TARGET_MIN_ELITE, int(len(passenger_stats) * MAX_ELITE_PERCENT))

    if len(df_elite) > max_elite_count:
        df_elite = df_elite.head(max_elite_count).copy()
        analytical_threshold = df_elite['passenger_rating_0_5'].min()
        method_used += f" + ограничено до топ-{len(df_elite)}"
        print(f"Элита сокращена до {len(df_elite)} пассажиров (лимит: {max_elite_count})")

print(f"\nИтоговый порог элиты: {analytical_threshold:.3f}")
print(f"Использованный метод: {method_used}")
print(f"\nЛог адаптации:")
for log_entry in method_log:
    print(f"{log_entry}")

# Статистика по опыту пассажиров
passengers_with_min_rides = (passenger_stats['rides_count'] >= ELITE_MIN_RIDES).sum()
print(f"\nКонтекст:")
print(f"Пассажиров с ≥{ELITE_MIN_RIDES} поездок: {passengers_with_min_rides}")
print(
    f"Их средний рейтинг: {passenger_stats[passenger_stats['rides_count'] >= ELITE_MIN_RIDES]['passenger_rating_0_5'].mean():.2f}")
print(
    f"Их медианный рейтинг: {passenger_stats[passenger_stats['rides_count'] >= ELITE_MIN_RIDES]['passenger_rating_0_5'].median():.2f}")
print(
    f"Макс. рейтинг среди них: {passenger_stats[passenger_stats['rides_count'] >= ELITE_MIN_RIDES]['passenger_rating_0_5'].max():.2f}")

# БЛОК 4: ВИЗУАЛИЗАЦИЯ
print("\nПостроение графика...")

ratings_sorted = passenger_stats['passenger_rating_0_5'].values
n_passengers = len(ratings_sorted)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Анализ элиты пассажиров (адаптивный порог)', fontsize=16, fontweight='bold')

# График 1: Ранжированный рейтинг
ax = axes[0, 0]
x = np.arange(1, n_passengers + 1)
ax.plot(x, ratings_sorted, linewidth=1.5, color='#2E86AB', label='Рейтинг пассажира', zorder=2)
ax.axhline(y=analytical_threshold, color='#E94F37', linestyle='--', linewidth=2.5,
           label=f'Адаптивный порог: {analytical_threshold:.2f}', zorder=3)
ax.axhline(y=passenger_stats['passenger_rating_0_5'].quantile(0.90), color='#7209B7', linestyle=':', linewidth=1)
if len(df_elite) > 0:
    elite_positions = passenger_stats[passenger_stats['passenger_id'].isin(df_elite['passenger_id'])].index
    for pos in elite_positions:
        ax.axvspan(pos, pos + 1, alpha=0.15, color='#E94F37')
ax.set_xlabel('Позиция в рейтинге (1 = лучший)', fontsize=10)
ax.set_ylabel('Рейтинг (0–5)', fontsize=10)
ax.set_title('Ранжированное распределение рейтинга пассажиров', fontsize=11, fontweight='bold')
ax.legend(fontsize=9, frameon=True)
ax.grid(alpha=0.3)
ax.set_xlim(1, n_passengers)

# График 2: Рейтинг vs Поездки
ax = axes[0, 1]
ax.scatter(passenger_stats['rides_count'], passenger_stats['passenger_rating_0_5'],
           c=passenger_stats['passenger_rating_0_5'], cmap='viridis', s=30, alpha=0.5, edgecolors='white',
           linewidth=0.3)
if len(df_elite) > 0:
    ax.scatter(df_elite['rides_count'], df_elite['passenger_rating_0_5'],
               c='#E94F37', s=100, edgecolors='black', linewidth=2,
               label=f'Элита ({len(df_elite)} пассажиров)', zorder=5)
ax.axhline(y=analytical_threshold, color='#E94F37', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(x=ELITE_MIN_RIDES, color='#F18F01', linestyle=':', linewidth=1.5, alpha=0.7,
           label=f'Мин. поездок: {ELITE_MIN_RIDES}')
ax.set_xlabel('Количество поездок (лог. шкала)', fontsize=10)
ax.set_ylabel('Рейтинг', fontsize=10)
ax.set_title('Рейтинг пассажира vs Частота использования', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xscale('log')

# График 3: Гистограмма
ax = axes[1, 0]
candidates = passenger_stats[passenger_stats['rides_count'] >= ELITE_MIN_RIDES]['passenger_rating_0_5']
if len(candidates) > 0:
    bins = min(20, len(candidates) // 3)
    ax.hist(candidates, bins=bins, edgecolor='white', color='#7209B7', alpha=0.7, label=f'≥{ELITE_MIN_RIDES} поездок')
    ax.axvline(x=analytical_threshold, color='#E94F37', linestyle='--', linewidth=2.5,
               label=f'Порог элиты: {analytical_threshold:.2f}')
    ax.set_xlabel('Рейтинг', fontsize=10)
    ax.set_ylabel('Количество пассажиров', fontsize=10)
    ax.set_title(f'Распределение рейтингов (≥{ELITE_MIN_RIDES} поездок)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
else:
    ax.text(0.5, 0.5, f'Нет пассажиров\nс ≥{ELITE_MIN_RIDES} поездок', transform=ax.transAxes, ha='center', va='center',
            fontsize=12)
    ax.set_title(f'Распределение (≥{ELITE_MIN_RIDES} поездок)', fontsize=11, fontweight='bold')

# График 4: Детали элиты
ax = axes[1, 1]
if len(df_elite) > 0:
    y_pos = np.arange(len(df_elite))
    ax.barh(y_pos, df_elite['passenger_rating_0_5'].values,
            color=plt.cm.viridis(df_elite['trust_coef'].values), edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"#{i + 1}" for i in range(len(df_elite))], fontsize=8)
    ax.set_xlabel('Рейтинг', fontsize=10)
    ax.set_ylabel('Позиция в элите', fontsize=10)
    ax.set_title('Рейтинг пассажиров элиты', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

    # Подписи с количеством поездок и долей отмен
    for i, (idx, row) in enumerate(df_elite.iterrows()):
        label = f"{int(row['rides_count'])} поездок, {row['cancellation_rate'] * 100:.0f}% отмен"
        ax.text(row['passenger_rating_0_5'] + 0.05, i, label, va='center', fontsize=7)
else:
    ax.text(0.5, 0.5, 'Элита пуста', transform=ax.transAxes, ha='center', va='center', fontsize=14)
    ax.set_title('Детали элиты', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('passenger_elite_adaptive_analysis.png', dpi=150, bbox_inches='tight')
print("График сохранен: 'passenger_elite_adaptive_analysis.png'")
plt.show()

# БЛОК 5: ФИНАЛЬНЫЙ ОТЧЕТ И ВЫВОД ДАТАФРЕЙМА
print("\n" + "=" * 80)
print("ЭТАП 4: ФИНАЛЬНЫЙ ОТЧЕТ")
print("=" * 80)

print(f"\nИтоговые критерии элиты:")
print(f"Рейтинг ≥ {analytical_threshold:.3f} ({method_used})")
print(f"Поездок ≥ {ELITE_MIN_RIDES}")

print(f"\nРезультаты отбора:")
print(f"Всего пассажиров в анализе: {len(passenger_stats)}")
print(f"Пассажиров с ≥{ELITE_MIN_RIDES} поездок: {passengers_with_min_rides}")
print(f"Пассажиров в элите: {len(df_elite)}")

if len(df_elite) > 0:
    pct = 100 * len(df_elite) / len(passenger_stats)
    print(f"Доля элиты: {pct:.2f}% от всех пассажиров")
    print(f"\nСтатистика элиты:")
    print(
        f"Средний рейтинг: {df_elite['passenger_rating_0_5'].mean():.2f} ± {df_elite['passenger_rating_0_5'].std():.2f}")
    print(f"Медиана рейтинга: {df_elite['passenger_rating_0_5'].median():.2f}")
    print(
        f"Диапазон: [{df_elite['passenger_rating_0_5'].min():.2f}, {df_elite['passenger_rating_0_5'].max():.2f}]")
    print(
        f"Среднее число поездок: {df_elite['rides_count'].mean():.0f} (медиана: {df_elite['rides_count'].median():.0f})")
    print(f"Средняя доля отмен: {df_elite['cancellation_rate'].mean() * 100:.1f}%")
    print(f"Средняя оценка от водителей: {df_elite['avg_passenger_rating'].mean():.2f}/5.0")
    print(f"Средний trust_coef: {df_elite['trust_coef'].mean():.3f}")

# ВЫВОД ПОЛНОГО ДАТАФРЕЙМА
result_columns = [
    'passenger_id', 'passenger_rating_0_5', 'rides_count',
    'cancellation_rate', 'no_show_rate', 'avg_passenger_rating', 'trust_coef'
]
if 'payment_issue_rate' in df_elite.columns:
    result_columns.insert(4, 'payment_issue_rate')

final_best_passengers_df = df_elite[result_columns].copy().reset_index(drop=True)

print("\n" + "=" * 80)
print("ПОЛНЫЙ ДАТАФРЕЙМ: ПАССАЖИРЫ ЭЛИТЫ")
print("=" * 80)

if len(final_best_passengers_df) > 0:
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(f"\nВсего записей: {len(final_best_passengers_df)}\n")
    print(final_best_passengers_df.to_string(index=True))
    pd.set_option('display.float_format', '{:.4f}'.format)
