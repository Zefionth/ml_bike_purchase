import pandas as pd
import os

# Создаём папку data если нет
os.makedirs('./data', exist_ok=True)

# 1. Загрузка данных
url = 'https://raw.githubusercontent.com/evgpat/edu_stepik_practical_ml/main/datasets/bike_buyers_clean.csv'
df = pd.read_csv(url)

print(f"Загружено данных: {df.shape}")
print(f"Колонки: {df.columns.tolist()}")
print(f"Распределение целевой переменной:\n{df['Purchased Bike'].value_counts()}")

# Сохраняем сырые данные
df.to_csv('./data/raw_bike_data.csv', index=False)

# 2. Очистка и предобработка
print(f"\nНачальный размер: {df.shape}")

# Пропуски в категориальных признаках
categorical_cols = ['Marital Status', 'Gender', 'Education', 'Occupation', 
                    'Home Owner', 'Commute Distance', 'Region']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

# Пропуски в числовых признаках
numerical_cols = ['Income', 'Children', 'Cars', 'Age']
for col in numerical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Удаление дубликатов
initial_len = len(df)
df = df.drop_duplicates()
print(f"Удалено дубликатов: {initial_len - len(df)}")

# Обработка выбросов
for col in numerical_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)

# Проверка на аномальные значения
df = df[df['Income'] >= 0]
df = df[(df['Age'] >= 18) & (df['Age'] <= 100)]
df = df[df['Children'] >= 0]
df = df[df['Cars'] >= 0]

# Сброс индекса
df = df.reset_index(drop=True)

# Сохраняем очищенные данные
df.to_csv('./data/df_clear.csv', index=False)

print(f"\nФинальный размер: {df.shape}")
print(f"Статистика по данным:\n{df.describe()}")
print("Данные загружены и очищены!")