# Объемная шпаргалка по Python-библиотекам для анализа данных: Pandas, Numpy, Matplotlib, Seaborn

## Pandas

### Импорт и основные объекты
```python
import pandas as pd
df = pd.DataFrame(data)                  # DataFrame
s = pd.Series([1,2,3])                   # Series
```

### Чтение и запись данных
```python
df = pd.read_csv('file.csv')             # CSV
df = pd.read_excel('file.xlsx')          # Excel
pd.read_json('file.json')                # JSON
pd.read_sql("SELECT * ...", conn)        # SQL

df.to_csv('out.csv')                     # В файл CSV
df.to_excel('out.xlsx')                  # Excel
df.to_json('out.json')                   # JSON
df.to_sql('table', conn)                 # SQL
```

### Обзор и информация
```python
df.head(), df.tail(), df.sample(5)       # Просмотр данных
df.shape, df.columns, df.index           # Размер, имена столбцов, индексы
df.info()                                # Информация о DataFrame
df.describe()                            # Описательная статистика
df.memory_usage(deep=True)               # Использование памяти
```

### Выборка и фильтрация
```python
df['col'], df.col                        # Колонка
df[['col1','col2']]                      # Несколько колонок
df.loc[5], df.iloc[5]                    # По индексу (метка/позиция)
df.loc[0:5,'col']                        # Срез
df[df['col'] > 5]                        # Фильтрация
(df['a']==1) & (df['b']<5)               # Логические условия
```

### Изменение и обновление данных
```python
df['new'] = value                        # Добавить столбец
df.drop('col', axis=1)                   # Удалить столбец
df.drop([0,2], axis=0)                   # Удалить строки
df.rename(columns={'old':'new'})         # Переименовать столбцы
df.rename(index={0:'first'})             # Переименовать индексы
df.set_index('col')                      # Индекс по колонке
df.reset_index()                         # Сбросить индекс
df.insert(1, 'newcol', values)           # Вставить столбец
df.replace({'a':1, 'b':2})               # Замена значений
df.apply(lambda x: x*2)                  # Применить функцию к колонке
```

### Работа с отсутствующими данными
```python
df.isnull(), df.notnull()                # Проверка на NaN
df.dropna(), df.dropna(axis=1)           # Удалить строки/столбцы с NaN
df.fillna(0)                             # Заполнить NaN значением
df.fillna({'col':0})                     # Заполнить по столбцам
df.interpolate()                         # Интерполяция NaN
```

### Группировка и агрегация
```python
df.groupby('col').sum()
df.groupby(['a','b'])['c'].mean()
df.groupby('col').agg(['sum','mean'])
df.pivot_table(values, index, columns)   # Сводная таблица
df.agg({'a':'sum','b':'mean'})           # Несколько агрегаций
df.value_counts(), df['col'].value_counts()   # Частоты
```

### Сортировка и индексация
```python
df.sort_values('col')
df.sort_values(['a','b'], ascending=[True,False])
df.sort_index()
df.nsmallest(3, 'col'), df.nlargest(3, 'col') # Топ по значениям
```

### Объединение и соединение данных
```python
pd.concat([df1, df2])                    # Объединить по строкам/столбцам
pd.merge(df1, df2, on='col')             # Соединение по колонке
pd.merge(df1, df2, left_on='a', right_on='b')  # Разные ключи
pd.merge(df1, df2, how='outer')          # outer join
join_df = df1.join(df2)                  # По индексам
```

### Работа с датой и временем
```python
df['date'] = pd.to_datetime(df['date'])
df['date'].dt.year, df['date'].dt.month  # Извлечь год, месяц
df['date'].dt.weekday, df['date'].dt.day_name() # День недели, имя дня
df['date'].dt.strftime('%d-%m-%Y')
df.set_index('date').resample('M').mean() # Ресемплинг по месяцам
```

### Статистика и вычисления
```python
df.corr(), df.cov(), df.min(), df.max(), df.sum(), df.mean(), df.median(), df.std(), df.quantile(0.95)
df['col'].unique(), df['col'].nunique(), df['col'].mode()
df.apply(np.log)
df.diff(), df.pct_change()
df.duplicated(), df.drop_duplicates()
```

### Сохранение, запись, копирование
```python
df.copy(), df.T                     # Копия, транспонирование
```

### Применение функций
```python
df['new'] = df['old'].map(lambda x: ...)
df.apply(lambda row: ..., axis=1)
df.applymap(lambda x: ..., na_action='ignore')  # Для всего DataFrame
```

### Категориальные данные
```python
df['cat'] = df['cat'].astype('category')
df['cat'].cat.codes, df['cat'].cat.categories
df['cat'].cat.add_categories(['new'])
df['cat'].cat.rename_categories({'old':'new'})
```


## Numpy

### Импорт и создание массивов
```python
import numpy as np
a = np.array([1, 2, 3])
b = np.zeros((3,4))
c = np.ones((2,2))
d = np.eye(3)               # Единичная матрица
f = np.full((2,3), 7)       # Заполнена 7
np.arange(0, 10, 2)         # Массив с шагом 2
np.linspace(0, 1, 5)        # 5 чисел от 0 до 1
np.random.rand(2,3), np.random.randn(3,3) # Случайные
np.random.randint(0,10,5)   # Случайные целые
```

### Основные операции с массивами
```python
b.shape, b.size, b.ndim, b.dtype  # Размер, количество элементов, измерения, тип
np.reshape(a, (1,3)), a.flatten(), b.T     # Изменение формы, плоский и транспонирование
np.concatenate([a, a]), np.vstack([a,a]), np.hstack([a,a]) # Объединение
np.split(a,2), np.array_split(a,3)  # Разделение
np.copy(a)                         # Копирование
```

### Индексация и срезы
```python
a[0], a[-1]
a[:2], a[::2]
a[1:3]
a[mask]                            # Булева индексация
b[0,1], b[:,0], b[1,:]
b[1:3, 2:4]
b[[0,2],[1,3]]
np.where(a>2)
np.argwhere(b==1)
```

### Математические и логические операции
```python
a+b, a-b, a*b, a/b
a**2, np.sqrt(a)
np.sum(a), np.mean(a), np.std(a)
np.min(a), np.max(a), np.median(a)
np.argsort(a)
np.unique(a), np.sort(a)
np.dot(a, a), np.cross(a, a)
np.cov(a), np.corrcoef(a)
np.exp(a), np.log(a), np.abs(a)
np.clip(a, 0, 1)
```

### Работа с NaN, Inf
```python
np.isnan(a), np.isfinite(a), np.isinf(a)
np.nanmean(a), np.nanstd(a)
a = np.nan_to_num(a)
```

## Matplotlib

### Основы построения
```python
import matplotlib.pyplot as plt
plt.plot([1,2,3],[4,5,6], label='Line')
plt.scatter(x, y, color='r')
plt.bar(['A','B'], [1,2])
plt.hist(data, bins=30)
plt.pie([10,20,30], labels=['A','B','C'])
plt.fill_between(x, y1, y2)
plt.errorbar(x, y, yerr=err)
```

### Настройки графика
```python
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Title')
plt.legend()
plt.grid(True)
plt.xlim(0,10)
plt.ylim(0,5)
plt.xticks(rotation=45)
plt.tight_layout()
```

### Сохранение и вывод
```python
plt.savefig('plot.png')
plt.show()
```

### Работа с несколькими графиками
```python
plt.subplot(2,1,1)
plt.subplot(221)
fig, ax = plt.subplots()
fig, axs = plt.subplots(2,2)
```

### Цвета и стили
```python
plt.style.use('ggplot')
plt.plot(x, y, color='red', linestyle='--', marker='o')
```

### Annotations и текст
```python
plt.text(2, 4, 'Point', fontsize=12)
plt.annotate('Max', xy=(x, y), xytext=(x+1, y+1), arrowprops=dict(facecolor='black'))
```

## Seaborn

### Импорт и основные графики
```python
import seaborn as sns
sns.set(style='whitegrid')
sns.lineplot(x='x', y='y', data=df)
sns.scatterplot(x='a', y='b', hue='c', data=df)
sns.barplot(x='label', y='val', data=df)
sns.countplot(x='label', data=df)
sns.histplot(data=df['col'], bins=20)
sns.boxplot(x='cat', y='val', data=df)
sns.violinplot(x='cat', y='val', data=df)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
sns.pairplot(df)
sns.jointplot(x='a', y='b', data=df)
sns.lmplot(x='x', y='y', data=df)
```

### Настройки и оформление
```python
sns.set_theme(style='darkgrid')
sns.set_palette('pastel')
sns.despine()
sns.set_context('paper')
```

### Фасетные и групповые графики
```python
g = sns.FacetGrid(df, col='cat', row='group')
g.map(sns.scatterplot, 'a', 'b')
g.add_legend()
g.savefig('facet.png')
```

### Цвет, размер, маркеры
```python
sns.scatterplot(x='a', y='b', data=df, hue='cat', size='val', style='group', palette='muted')
```

---

**Полезные паттерны**:
- Для изучения: документация, методы help(), dir(), поиск по StackOverflow, тестирование в Jupyter
- Все короткие паттерны удобно разместить в отдельном файле.
- Pandas display options: pd.options.display.max_rows, pd.options.display.float_format

**Рекомендации**:
- Для визуализации: plt.figure(figsize=(10,5)), sns.set(rc={'figure.figsize':(12,6)})
- Для анализа: df.info(), np.nanmean(a), sns.heatmap(df.corr())

Документация:
- Pandas: https://pandas.pydata.org/docs/
- Numpy: https://numpy.org/doc/
- Matplotlib: https://matplotlib.org/
- Seaborn: https://seaborn.pydata.org/

Шпаргалка оформлена с примерами, комментариями и готова к сохранению/копированию.