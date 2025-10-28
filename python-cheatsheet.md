# Объемная шпаргалка по Python

## Запуск и работа с интерпретатором
```bash
python                          # Запуск Python
python script.py                # Выполнить скрипт
python -i script.py             # После выполнения открыть интерактив
python3                         # Запуск Python 3
pip install package             # Установить пакет
pip list                        # Список установленных пакетов
pip freeze > requirements.txt   # Экспорт зависимостей
pip install -r requirements.txt # Установка из файла
```

## Структура программы
```python
# Однострочный комментарий
# Многострочный комментарий — использовать ''' или """

if cond:
    pass  # Условный оператор

for i in range(5):
    print(i)

while x < 10:
    x += 1

try:
    ...
except Exception as e:
    print(e)
finally:
    ...

# Функция
def func(arg1, arg2=5):
    return arg1 + arg2

# Класс
class MyClass:
    def __init__(self, value):
        self.value = value
    def method(self):
        print(self.value)
```

## Типы данных
```python
x = 10                 # int
f = 2.5                # float
b = True               # bool
s = "строка"           # str
l = [1, 2, 3]          # list
t = (1, 2, 3)          # tuple
d = {'a': 1, 'b': 2}   # dict
s = {1, 2, 3}          # set
n = None               # NoneType
```

## Операции и выражения
```python
+, -, *, /, //, %      # Арифметика
**,                    # Возведение в степень
==, !=, >, <, >=, <=   # Сравнения
and, or, not           # Логика
in, not in             # Вхождение
is, is not             # Сравнение ссылок
[]                     # Индексация
[:]                    # Срезы
```

## Работа со списками, генераторы
```python
l = [1, 2, 3]
l.append(4)
l.extend([5,6])
l.insert(0, 0)
l.pop()         # По умолчанию последний
l.remove(1)     # По значению
l.sort()
l.reverse()
l2 = [x*2 for x in l]           # Список-генератор
l3 = [x for x in l if x > 2]     # С фильтрацией
```

## Работа с dict, set, tuple
```python
d = {'a': 1, 'b': 2}
d['c'] = 3
d.get('x', 0)
d.keys(), d.values(), d.items()
d.pop('a')
d.update({'d': 4})
del d['b']
s = set([1, 2, 2, 3])             # {1,2,3}
s.add(4)
s.remove(1)
s.clear()
s.union({5,6})
s.intersection({2,3,4})
s.difference({3})
t = (1, 2, 3)
t.count(2)
t.index(1)
```

## Работа со строками
```python
s = "Привет"
s.upper()
s.lower()
s.title()
s.capitalize()
s.strip()
s.replace('и', 'е')
s.split(',')
'-'.join(['a','b'])
s.find('в')
s.startswith('П')
s.endswith('т')
```

## Форматирование строк
```python
name = "Вася"
af = f"Привет, {name}!"
ss = "{} {}".format('a', 'b')
```

## Работа с файлами
```python
with open('file.txt', 'r') as f:
    text = f.read()
with open('file.txt', 'w') as f:
    f.write("текст")
with open('file.txt', 'a') as f:
    f.write("добавить")
for line in open('file.txt'):
    print(line.strip())
```

## Основные стандартные библиотеки
```python
import math
math.sqrt(16)                   # Извлечение корня
math.pi                         # Число pi
math.ceil(2.1)                  # Округление вверх

import random
random.randint(1,100)
random.choice([1,2,3])
random.sample(range(100), 5)

import datetime
datetime.datetime.now()
datetime.date.today()
import time
time.sleep(1)                   # Пауза 1 сек

defaultdict, Counter, namedtuple (collections)

import os
os.listdir('.')
os.getcwd()
os.makedirs('dir')
os.path.exists('f')
os.remove('f')

import sys
sys.argv                        # Аргументы скрипта
sys.exit(1)                     # Завершить скрипт с кодом 1

import json
data = json.loads('{"a":1}')
s = json.dumps({'a':1})

import re
re.match('a.', 'abc')
re.search('b', 'abc')
re.findall(r'\w+', 'a b')
```

## Модули и пакеты
```python
# Импорт всего модуля
import math

# Импорт подмодуля
from collections import Counter

# Импорт конкретной функции
from os import path
```

## Функции высшего порядка, lambda, map, filter
```python
f = lambda x: x + 2
l2 = list(map(lambda x: x**2, [1,2,3]))
l3 = list(filter(lambda x: x>2, [1,2,3]))
```

## Исключения
```python
try:
    ...
except ZeroDivisionError:
    ...
except Exception as e:
    print(e)
else:
    ...
finally:
    ...
raise ValueError("ошибка")
assert x > 0, "x должен быть положительным"
```

## Итераторы и генераторы
```python
for i in [1,2,3]:
    print(i)

gen = (x*2 for x in range(3))
next(gen)

def genfunc():
    for i in range(3):
        yield i*2
for k in genfunc():
    print(k)
```

## ООП
```python
class Animal:
    def __init__(self, name):
        self.name = name
    def speak(self):
        print(f"Я {self.name}")

class Dog(Animal):
    def speak(self):
        print(f"Гав, я {self.name}")

obj = Dog("Шарик")
obj.speak()
```

## List comprehension, dict/set comprehension
```python
[x*2 for x in range(5)]
{str(x): x for x in range(5)}
{x for x in range(10) if x%2==0}
```

## Типизация (примеры с typing)
```python
from typing import List, Dict, Tuple, Optional

def func(l: List[int]) -> int:
    return sum(l)
```

## Jupyter/REPL команды
```python
%run script.py         # Запустить скрипт
%timeit x = [i for i in range(10)]
%lsmagic               # Список магических команд
```

## Виртуальное окружение
```bash
python -m venv venv              # Создать виртуальное окружение
source venv/bin/activate         # Активировать (Linux/Mac)
venv\Scripts\activate            # Активировать (Windows)
deactivate                       # Деактивировать
```

## Полезные паттерны
```python
# Однострочный swap переменных
a, b = b, a
# Одновременный unpack
x, y, z = (1, 2, 3)
# Функция с переменным количеством аргументов
def foo(*args, **kwargs):
    print(args, kwargs)
# enumerate для индекса
for idx, val in enumerate(['a','b']):
    print(idx, val)
# Сортировка по ключу
sorted(d.items(), key=lambda x: x[1])
# zip для объединения списков
for a, b in zip([1,2], [10,20]):
    ...
```

---

**Для изучения**: `help`, `dir`, `type`, `vars`, `isinstance`, `id`, встроенные функции и документация https://docs.python.org/3/

**Совет**: все короткие паттерны удобно разместить в отдельном файле для быстрого поиска.