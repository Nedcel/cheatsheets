# Полезные команды и шпаргалка по SQL для PostgreSQL

## Основы работы с базой данных

### Подключение
```sql
psql -U <user> -d <database>              -- Подключиться к базе через терминал
psql -h <host> -p <port> -U <user> <db>   -- С параметрами хоста и порта
```

### Информация о базе
```sql
\l                                      -- Список баз данных
\c <database>                            -- Переключиться на другую базу
\dt                                     -- Список таблиц
\d <table>                               -- Структура таблицы
\d+ <table>                              -- Структура + доп. информация
\df                                      -- Список функций
\dv                                      -- Список представлений
\dn                                      -- Список схем
\du                                      -- Список ролей
\x                                       -- Формат вывода: on/off
\?                                       -- Справка по командам
```

## Основные SQL-запросы

### SELECT — выборка данных
```sql
SELECT * FROM table;                      -- Все данные из таблицы
SELECT column1, column2 FROM table;       -- Указанные столбцы
SELECT DISTINCT column FROM table;        -- Уникальные значения
SELECT * FROM table LIMIT 10;             -- Первые 10 строк
SELECT * FROM table OFFSET 5 LIMIT 10;    -- С 6 по 15 строки
SELECT * FROM table WHERE col = 'value';  -- Условие
SELECT * FROM table WHERE col1 > 5 AND col2 != 'str'; -- Сложные условия
SELECT * FROM table WHERE col IS NULL;    -- Проверка на NULL
SELECT * FROM table WHERE col IN ('a','b'); -- Вхождение в список
SELECT * FROM table WHERE col LIKE 'A%';  -- Шаблон поиска
SELECT * FROM table ORDER BY col DESC;    -- Сортировка
SELECT COUNT(*) FROM table;               -- Количество строк
SELECT MAX(col), MIN(col) FROM table;     -- Максимум, минимум
SELECT AVG(col), SUM(col) FROM table;     -- Среднее, сумма
SELECT col1, COUNT(*) FROM table GROUP BY col1; -- Группировка
SELECT col, COUNT(*) FROM table GROUP BY col HAVING COUNT(*) > 1; -- Фильтр по группе
SELECT col1, col2 FROM table1 JOIN table2 ON table1.key = table2.key; -- JOIN
SELECT * FROM table1 LEFT JOIN table2 ON ...;     -- Левое соединение
SELECT * FROM table1 RIGHT JOIN table2 ON ...;    -- Правое соединение
SELECT * FROM table1 FULL OUTER JOIN table2 ON ...; -- Полное соединение
SELECT ... FROM table UNION SELECT ...;           -- Объединение результатов
SELECT ... FROM table EXCEPT SELECT ...;          -- Исключение
SELECT ... FROM table INTERSECT SELECT ...;       -- Пересечение
WITH tmp AS (SELECT ...) SELECT * FROM tmp;       -- CTE выражения
```

### Вставка/изменение/удаление данных
```sql
INSERT INTO table (col1, col2) VALUES ('val1', 'val2');
INSERT INTO table DEFAULT VALUES;                       -- Вставка строки по умолчанию
INSERT INTO table (col1, col2) SELECT col3, col4 FROM table2; -- Вставка из запроса

UPDATE table SET col = 'val' WHERE cond;
UPDATE table SET col1 = expr1, col2 = expr2 WHERE cond;

DELETE FROM table WHERE cond;
TRUNCATE table;                                         -- Очистить всю таблицу
```

### Работа со структурами

#### Таблицы
```sql
CREATE TABLE table (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT DEFAULT 18,
    created_at TIMESTAMP DEFAULT NOW()
);

DROP TABLE table;                                      -- Удалить таблицу
ALTER TABLE table ADD COLUMN col INT;
ALTER TABLE table DROP COLUMN col;
ALTER TABLE table RENAME COLUMN old TO new;
ALTER TABLE table ALTER COLUMN col TYPE TEXT;
ALTER TABLE table SET SCHEMA другое_имя_схемы;
```

#### Индексы
```sql
CREATE INDEX idx_name ON table (col);
CREATE UNIQUE INDEX uniq_name ON table (col);
DROP INDEX idx_name;
```

#### Внешние ключи и ограничения
```sql
ALTER TABLE table ADD CONSTRAINT fk_name FOREIGN KEY (col) REFERENCES other_table (col);
ALTER TABLE table DROP CONSTRAINT fk_name;
ALTER TABLE table ADD CONSTRAINT unique_name UNIQUE (col);
ALTER TABLE table ADD CONSTRAINT check_age CHECK (age > 0);
```

#### Последовательности
```sql
CREATE SEQUENCE seq_name START 1 INCREMENT 1;
SELECT nextval('seq_name');
ALTER SEQUENCE seq_name RESTART WITH 1;
DROP SEQUENCE seq_name;
```

### Представления и функции

#### Представления
```sql
CREATE VIEW view_name AS SELECT ...;
DROP VIEW view_name;
```

#### Хранимые функции
```sql
CREATE FUNCTION func_name(arg1 INT) RETURNS INT AS $$
BEGIN
    RETURN arg1 * 2;
END;
$$ LANGUAGE plpgsql;

DROP FUNCTION func_name(INT);
```

#### Триггеры
```sql
CREATE TRIGGER trig_name AFTER INSERT ON table FOR EACH ROW EXECUTE PROCEDURE func_name();
DROP TRIGGER trig_name ON table;
```

## Пользователи и права
```sql
CREATE USER username WITH PASSWORD 'pass';
ALTER USER username WITH PASSWORD 'newpass';
DROP USER username;
GRANT ALL PRIVILEGES ON DATABASE db TO username;
GRANT SELECT, INSERT ON table TO username;
REVOKE INSERT ON table FROM username;
```

## Системные команды и диагностика
```sql
SELECT version();                                 -- Версия PostgreSQL
SELECT current_database();                         -- Текущая БД
SELECT current_user;                              -- Текущий пользователь
SELECT now();                                     -- Текущее время
SELECT * FROM pg_tables;                          -- Таблицы в БД
SELECT * FROM pg_indexes;                         -- Индексы
SELECT * FROM pg_views;                           -- Представления
SELECT * FROM pg_stat_activity;                   -- Текущие подключения
SELECT * FROM information_schema.tables;          -- Инфо о таблицах
SELECT * FROM information_schema.columns WHERE table_name='table'; -- Инфо о столбцах
```

## Работа с транзакциями
```sql
BEGIN;                                            -- Начать транзакцию
SAVEPOINT svpt;                                   -- Точка сохранения
ROLLBACK TO svpt;                                 -- Откат к savepoint
COMMIT;                                           -- Подтвердить изменения
ROLLBACK;                                         -- Откатить транзакцию
```

## Утилиты и советы

### Экспорт и импорт
```bash
pg_dump dbname > dump.sql               # Export
pg_restore -d dbname dump.sql           # Import
psql -U user -d dbname -f file.sql      # Выполнить SQL скрипт
COPY table TO '/tmp/data.csv' WITH CSV; # Экспорт таблицы в CSV
COPY table FROM '/tmp/data.csv' WITH CSV; # Импорт из CSV
```

### Диагностика и оптимизация
```sql
EXPLAIN SELECT ...;                     -- План выполнения запроса
EXPLAIN ANALYZE SELECT ...;             -- Реальный план + затраты
VACUUM;                                 -- Очистка "мертвых" строк
VACUUM FULL;                            -- Полная очистка и сжатие
REINDEX TABLE table;                    -- Перестроить индексы
ANALYZE table;                          -- Сбор статистики
```

### Прочее
```sql
SELECT pg_size_pretty(pg_database_size('dbname'));  -- Размер БД
SELECT pg_size_pretty(pg_total_relation_size('table')); -- Размер таблицы
SELECT relname, n_live_tup FROM pg_stat_user_tables; -- Число строк в таблицах
```

---

**Дополнительные полезные ресурсы**:
- Используйте \? в psql для справки по командам
- DETAIL и EXPLAIN ANALYZE — незаменимы для оптимизации запросов
- pgAdmin, DataGrip — удобные GUI-инструменты для работы с PostgreSQL

**Для изучения**: документация https://www.postgresql.org/docs/, команда `man psql`, расширения `pg_stat_statements`, `uuid-ossp`, `pgcrypto`.

Шпаргалка оформлена для удобного копирования и постоянного использования.