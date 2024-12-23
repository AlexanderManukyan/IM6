import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = {
    '2008': [104.8, 111.3, 105.9, 101.0, 107.6, 99.7, 116.7, 104.0, 103.0, 103.1, 106.0, 106.1, 104.0, 107.8, 105.5],
    '2009': [95.9, 102.1, 95.3, 95.6, 98.2, 97.7, 95.2, 94.3, 98.5, 95.0, 96.7, 90.8, 97.1, 96.5, 101.0],
    '2010': [103.2, 110.0, 105.3, 106.4, 101.7, 100.6, 111.4, 105.0, 102.0, 104.0, 107.2, 104.9, 103.9, 106.3, 96.7],
    '2011': [103.3, 107.7, 107.6, 102.2, 111.4, 98.8, 113.2, 105.7, 107.0, 104.9, 109.7, 112.4, 107.4, 104.8, 111.9],
    '2012': [103.8, 103.5, 108.0, 102.7, 109.1, 97.2, 110.3, 104.8, 107.2, 102.3, 104.5, 103.4, 104.8, 106.0, 108.8]
}

# Преобразуем данные в DataFrame
df = pd.DataFrame(data)

# Печатаем структуру данных
print("Исходная таблица данных:")
print(df)

# Месяцы для каждого года
years = list(df.columns)
quarters = ['Q1', 'Q2', 'Q3', 'Q4']

# Создаем временной ряд для каждого квартала (разделяем на 4 квартала)
data_reshaped = []
for year in years:
    year_values = df[year].values
    # Разделим данные на 4 части
    quarters_values = np.array_split(year_values, 4)
    for q, value in zip(quarters, quarters_values):
        data_reshaped.append([year, q, value[0]])  # Используем первое значение в каждом квартале

# Создаем DataFrame с MultiIndex для кварталов и годов
reshaped_df = pd.DataFrame(data_reshaped, columns=['Year', 'Quarter', 'Value'])

# Печатаем результат
print("\nВременной ряд с кварталами:")
print(reshaped_df)

# Функция для реализации модели Хольта-Уинтерса
def holt_winters(ts, alpha, beta, gamma, season_length, forecast_periods):
    # Инициализация
    n = len(ts)
    level = np.zeros(n)
    trend = np.zeros(n)
    seasonal = np.zeros(season_length)
    forecast = np.zeros(forecast_periods)

    # Первоначальные значения уровня и тренда
    level[0] = ts[0]  # Начальный уровень
    trend[0] = ts[1] - ts[0]  # Начальный тренд (разница между первым и вторым значением)
    
    # Инициализация сезонности
    seasonal[:season_length] = ts[:season_length] / np.mean(ts[:season_length])

    # Применение алгоритма Хольта-Уинтерса
    for t in range(1, n):
        level[t] = alpha * (ts[t] / seasonal[t % season_length]) + (1 - alpha) * (level[t - 1] + trend[t - 1])
        trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
        seasonal[t % season_length] = gamma * (ts[t] / level[t]) + (1 - gamma) * seasonal[t % season_length]
    
    # Прогноз на будущие периоды
    for t in range(n, n + forecast_periods):
        forecast[t - n] = (level[n - 1] + (t - n + 1) * trend[n - 1]) * seasonal[t % season_length]
    
    return level, trend, seasonal, forecast

# Пример временного ряда (извлекаем данные для всех кварталов, например, для 2008 года)
ts = reshaped_df[reshaped_df['Year'] == '2008']['Value'].values

# Параметры сглаживания
alpha = 0.5  # Сглаживание уровня
beta = 0.5   # Сглаживание тренда
gamma = 0.5  # Сглаживание сезонности
season_length = 4  # Длительность сезона (квартал)
forecast_periods = 4  # Количество прогнозных периодов

# Применение модели Хольта-Уинтерса
level, trend, seasonal, forecast = holt_winters(ts, alpha, beta, gamma, season_length, forecast_periods)

# Вывод результатов
print("\nКоэффициенты парной регрессии (уровень, тренд):")
print(f"Уровень: {level[-1]}, Тренд: {trend[-1]}")
print("Коэффициенты сезонности:")
print(seasonal)

print("\nПрогноз на следующие 4 периода:")
print(forecast)

# Визуализация
plt.figure(figsize=(10, 6))
plt.plot(ts, label="Исходные данные", color="blue")
plt.plot(range(len(ts), len(ts) + forecast_periods), forecast, label="Прогноз", color="red")
plt.title("Прогнозирование с использованием модели Хольта-Уинтерса")
plt.legend()
plt.show()
