# Полезные команды Git и Bash

## Git - Базовые операции

### Первый акцес

# Создать SSH ключ (замените email на свой)
ssh-keygen -t ed25519 -C "your_email@example.com"


# Запустить SSH-агент
eval "$(ssh-agent -s)"

# Добавить ключ в агент
ssh-add ~/.ssh/id_ed25519

# Windows
cat ~/.ssh/id_ed25519.pub | clip

# Или просто посмотреть и скопировать вручную
cat ~/.ssh/id_ed25519.pub

ssh -T git@github.com
# Должен быть вывод: "Hi username! You've successfully authenticated..."

# Настройка пользователя (замените на свои данные)
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"

# Создать папку проекта и перейти в нее
mkdir my-project
cd my-project

# 1. Инициализировать локальный репозиторий
git init

# 2. Создать файлы проекта (пример)
echo "# My Project" > README.md
# Добавьте ваши файлы в папку

# 3. Добавить все файлы в staging
git add .

# 4. Сделать первый коммит
git commit -m "Initial commit"

# 5. Добавить удаленный репозиторий (ИСПОЛЬЗУЙТЕ SSH!)
git remote add origin git@github.com:username/my-project.git

# 6. Переименовать ветку в main
git branch -M main

# 7. Запушить в репозиторий
git push -u origin main

# Проверить статус
git status

# Проверить удаленные репозитории
git remote -v

# Проверить историю коммитов
git log --oneline


### Инициализация и клонирование
```bash
git init                          # Инициализировать новый репозиторий
git clone <url>                   # Клонировать репозиторий
git clone --depth 1 <url>         # Клонировать только последний коммит (shallow clone)
```

### Состояние и информация
```bash
git status                        # Показать статус рабочей директории
git status -s                     # Краткий статус
git log                           # История коммитов
git log --oneline                 # Компактный лог
git log --graph --oneline --all   # Граф веток
git log -p                        # Лог с изменениями
git log --author="name"           # Коммиты автора
git log --since="2 weeks ago"     # Коммиты за период
git show <commit>                 # Показать коммит
git diff                          # Изменения в рабочей директории
git diff --staged                 # Изменения в staging area
git diff <branch1> <branch2>      # Разница между ветками
```

### Добавление и коммиты
```bash
git add <file>                    # Добавить файл в staging
git add .                         # Добавить все изменения
git add -A                        # Добавить все (включая удаления)
git add -p                        # Интерактивное добавление
git commit -m "message"           # Создать коммит
git commit -am "message"          # Добавить и закоммитить tracked файлы
git commit --amend                # Изменить последний коммит
git commit --amend --no-edit      # Добавить в последний коммит без изменения сообщения
```

## Git - Ветки и слияния

### Управление ветками
```bash
git branch                        # Список веток
git branch -a                     # Все ветки (включая remote)
git branch <name>                 # Создать ветку
git branch -d <name>              # Удалить ветку
git branch -D <name>              # Принудительно удалить ветку
git branch -m <old> <new>         # Переименовать ветку
git checkout <branch>             # Переключиться на ветку
git checkout -b <branch>          # Создать и переключиться
git switch <branch>               # Переключиться (новый синтаксис)
git switch -c <branch>            # Создать и переключиться (новый)
```

### Слияние и rebase
```bash
git merge <branch>                # Слить ветку
git merge --no-ff <branch>        # Слить с созданием merge commit
git merge --squash <branch>       # Слить с объединением коммитов
git rebase <branch>               # Rebase на ветку
git rebase -i HEAD~3              # Интерактивный rebase последних 3 коммитов
git rebase --continue             # Продолжить rebase
git rebase --abort                # Отменить rebase
```

## Git - Удаленные репозитории

### Работа с remote
```bash
git remote -v                     # Список удаленных репозиториев
git remote add <name> <url>       # Добавить remote
git remote remove <name>          # Удалить remote
git remote rename <old> <new>     # Переименовать remote
git remote show <name>            # Информация о remote
```

### Push и Pull
```bash
git fetch                         # Получить изменения без слияния
git fetch --all                   # Получить из всех remote
git pull                          # Получить и слить изменения
git pull --rebase                 # Pull с rebase вместо merge
git push                          # Отправить изменения
git push -u origin <branch>       # Push и установить upstream
git push --force                  # Принудительный push (осторожно!)
git push --force-with-lease       # Безопасный force push
git push --all                    # Push всех веток
git push --tags                   # Push всех тегов
git push origin --delete <branch> # Удалить remote ветку
```

## Git - Отмена изменений

### Откат и reset
```bash
git restore <file>                # Отменить изменения в файле
git restore --staged <file>       # Убрать файл из staging
git checkout -- <file>            # Отменить изменения (старый синтаксис)
git reset <file>                  # Убрать из staging
git reset --soft HEAD~1           # Отменить коммит, оставить изменения в staging
git reset --mixed HEAD~1          # Отменить коммит, оставить в рабочей директории
git reset --hard HEAD~1           # Удалить коммит и все изменения
git reset --hard origin/<branch>  # Сбросить до состояния remote
git clean -n                      # Показать неотслеживаемые файлы
git clean -f                      # Удалить неотслеживаемые файлы
git clean -fd                     # Удалить неотслеживаемые файлы и директории
```

### Revert
```bash
git revert <commit>               # Создать коммит, отменяющий изменения
git revert HEAD                   # Отменить последний коммит
git revert --no-commit <commit>   # Revert без автоматического коммита
```

## Git - Stash

### Временное сохранение
```bash
git stash                         # Сохранить изменения
git stash save "message"          # Stash с сообщением
git stash list                    # Список stash
git stash show                    # Показать последний stash
git stash show -p                 # Показать изменения в stash
git stash apply                   # Применить последний stash
git stash apply stash@{n}         # Применить конкретный stash
git stash pop                     # Применить и удалить stash
git stash drop                    # Удалить последний stash
git stash drop stash@{n}          # Удалить конкретный stash
git stash clear                   # Удалить все stash
git stash branch <name>           # Создать ветку из stash
```

## Git - Теги

### Управление тегами
```bash
git tag                           # Список тегов
git tag <name>                    # Создать lightweight тег
git tag -a <name> -m "message"    # Создать annotated тег
git tag -a <name> <commit>        # Тег для конкретного коммита
git show <tag>                    # Показать тег
git push origin <tag>             # Push тега
git push --tags                   # Push всех тегов
git tag -d <name>                 # Удалить локальный тег
git push origin --delete <tag>    # Удалить remote тег
```

## Git - Поиск и отладка

### Поиск
```bash
git grep "pattern"                # Поиск в файлах
git grep -n "pattern"             # Поиск с номерами строк
git log --grep="pattern"          # Поиск в сообщениях коммитов
git log -S "code"                 # Поиск по содержимому изменений
git blame <file>                  # Кто изменил каждую строку
git blame -L 10,20 <file>         # Blame для строк 10-20
```

### Отладка
```bash
git bisect start                  # Начать бинарный поиск бага
git bisect bad                    # Отметить текущий коммит как плохой
git bisect good <commit>          # Отметить коммит как хороший
git bisect reset                  # Закончить bisect
```

## Git - Конфигурация

### Настройки
```bash
git config --global user.name "Name"           # Установить имя
git config --global user.email "email"         # Установить email
git config --global core.editor "vim"          # Установить редактор
git config --global alias.st status            # Создать алиас
git config --global alias.co checkout          # Алиас checkout
git config --global alias.br branch            # Алиас branch
git config --global alias.ci commit            # Алиас commit
git config --global alias.unstage 'reset HEAD' # Алиас unstage
git config --list                              # Показать конфигурацию
git config --global --list                     # Глобальная конфигурация
git config --local --list                      # Локальная конфигурация
```

### Полезные настройки
```bash
git config --global color.ui auto              # Цветной вывод
git config --global core.autocrlf input        # Обработка переносов строк (Linux/Mac)
git config --global core.autocrlf true         # Обработка переносов строк (Windows)
git config --global pull.rebase true           # Pull с rebase по умолчанию
git config --global init.defaultBranch main    # Имя ветки по умолчанию
```

## Git - Продвинутые команды

### Cherry-pick и reflog
```bash
git cherry-pick <commit>          # Применить коммит из другой ветки
git cherry-pick <commit1>..<commitN> # Cherry-pick диапазона
git reflog                        # История всех изменений HEAD
git reflog show <branch>          # Reflog для ветки
git checkout HEAD@{n}             # Вернуться к состоянию из reflog
```

### Submodules
```bash
git submodule add <url>           # Добавить submodule
git submodule init                # Инициализировать submodules
git submodule update              # Обновить submodules
git submodule update --remote     # Обновить до последней версии
git clone --recurse-submodules <url> # Клонировать с submodules
```

### Worktree
```bash
git worktree add <path> <branch>  # Создать новую рабочую директорию
git worktree list                 # Список worktrees
git worktree remove <path>        # Удалить worktree
```

## Bash - Навигация

### Перемещение по директориям
```bash
cd <dir>                          # Перейти в директорию
cd ..                             # Подняться на уровень выше
cd ~                              # Перейти в домашнюю директорию
cd -                              # Вернуться в предыдущую директорию
pwd                               # Показать текущую директорию
pushd <dir>                       # Перейти и сохранить в стек
popd                              # Вернуться из стека
dirs                              # Показать стек директорий
```

## Bash - Работа с файлами и директориями

### Просмотр содержимого
```bash
ls                                # Список файлов
ls -la                            # Подробный список (включая скрытые)
ls -lh                            # Список с человекочитаемыми размерами
ls -lt                            # Сортировка по времени
ls -lS                            # Сортировка по размеру
tree                              # Дерево директорий
tree -L 2                         # Дерево с глубиной 2
cat <file>                        # Вывести содержимое файла
less <file>                       # Просмотр файла с пагинацией
head <file>                       # Первые 10 строк
head -n 20 <file>                 # Первые 20 строк
tail <file>                       # Последние 10 строк
tail -n 20 <file>                 # Последние 20 строк
tail -f <file>                    # Следить за изменениями файла
wc <file>                         # Подсчет строк, слов, байтов
wc -l <file>                      # Только строки
```

### Создание и удаление
```bash
touch <file>                      # Создать пустой файл
mkdir <dir>                       # Создать директорию
mkdir -p <path/to/dir>            # Создать с родительскими директориями
rm <file>                         # Удалить файл
rm -f <file>                      # Принудительно удалить
rm -r <dir>                       # Удалить директорию рекурсивно
rm -rf <dir>                      # Принудительно удалить директорию
rmdir <dir>                       # Удалить пустую директорию
```

### Копирование и перемещение
```bash
cp <src> <dst>                    # Копировать файл
cp -r <src> <dst>                 # Копировать директорию
cp -a <src> <dst>                 # Копировать с сохранением атрибутов
cp -u <src> <dst>                 # Копировать только новые файлы
mv <src> <dst>                    # Переместить/переименовать
mv -i <src> <dst>                 # С подтверждением перезаписи
```

### Поиск файлов
```bash
find . -name "*.txt"              # Найти по имени
find . -type f                    # Найти только файлы
find . -type d                    # Найти только директории
find . -mtime -7                  # Файлы изменённые за последние 7 дней
find . -size +100M                # Файлы больше 100 МБ
find . -name "*.log" -delete      # Найти и удалить
find . -name "*.txt" -exec cat {} \; # Выполнить команду для найденных
locate <file>                     # Быстрый поиск по базе
which <command>                   # Путь к команде
whereis <command>                 # Путь к бинарнику, исходникам, man
```

## Bash - Работа с текстом

### Поиск и фильтрация
```bash
grep "pattern" <file>             # Поиск в файле
grep -r "pattern" <dir>           # Рекурсивный поиск
grep -i "pattern" <file>          # Без учета регистра
grep -v "pattern" <file>          # Инвертировать поиск
grep -n "pattern" <file>          # С номерами строк
grep -c "pattern" <file>          # Подсчет совпадений
grep -A 3 "pattern" <file>        # +3 строки после
grep -B 3 "pattern" <file>        # +3 строки до
grep -C 3 "pattern" <file>        # +3 строки до и после
```

### Обработка текста
```bash
sed 's/old/new/' <file>           # Заменить первое вхождение
sed 's/old/new/g' <file>          # Заменить все вхождения
sed -i 's/old/new/g' <file>       # Изменить файл на месте
sed -n '10,20p' <file>            # Вывести строки 10-20
awk '{print $1}' <file>           # Вывести первую колонку
awk '{print $NF}' <file>          # Вывести последнюю колонку
awk -F',' '{print $1}' <file>     # С разделителем запятая
cut -d',' -f1 <file>              # Вырезать первое поле (CSV)
sort <file>                       # Сортировать
sort -r <file>                    # Обратная сортировка
sort -n <file>                    # Числовая сортировка
sort -u <file>                    # Сортировать и удалить дубликаты
uniq <file>                       # Удалить соседние дубликаты
uniq -c <file>                    # Подсчитать дубликаты
tr 'a-z' 'A-Z' < <file>           # Преобразовать в верхний регистр
tr -d ' ' < <file>                # Удалить пробелы
```

## Bash - Перенаправление и конвейеры

### Потоки ввода/вывода
```bash
command > file                    # Перенаправить stdout в файл
command >> file                   # Добавить stdout в файл
command 2> file                   # Перенаправить stderr
command &> file                   # Перенаправить stdout и stderr
command < file                    # Использовать файл как stdin
command1 | command2               # Конвейер (pipe)
command | tee file                # Вывести и сохранить в файл
command 2>&1                      # Перенаправить stderr в stdout
```

## Bash - Процессы и система

### Управление процессами
```bash
ps                                # Список процессов
ps aux                            # Подробный список всех процессов
ps aux | grep <name>              # Найти процесс
top                               # Мониторинг процессов в реальном времени
htop                              # Улучшенный top (если установлен)
kill <pid>                        # Завершить процесс
kill -9 <pid>                     # Принудительно завершить
killall <name>                    # Завершить все процессы с именем
pkill <pattern>                   # Завершить по паттерну
bg                                # Продолжить в фоне
fg                                # Вернуть на передний план
jobs                              # Список фоновых задач
command &                         # Запустить в фоне
nohup command &                   # Запустить с игнорированием SIGHUP
```

### Информация о системе
```bash
uname -a                          # Информация о системе
hostname                          # Имя хоста
uptime                            # Время работы системы
date                              # Текущая дата и время
cal                               # Календарь
whoami                            # Текущий пользователь
who                               # Кто залогинен
w                                 # Кто залогинен и что делает
id                                # ID пользователя и группы
groups                            # Группы пользователя
```

### Диски и память
```bash
df -h                             # Использование дисков
du -h <dir>                       # Размер директории
du -sh <dir>                      # Итоговый размер
du -h --max-depth=1               # Размеры поддиректорий
free -h                           # Использование памяти
lsblk                             # Список блочных устройств
mount                             # Список смонтированных FS
fdisk -l                          # Список дисков (требует root)
```

## Bash - Сеть

### Сетевые команды
```bash
ping <host>                       # Проверить доступность хоста
ping -c 4 <host>                  # Ping 4 раза
curl <url>                        # Загрузить URL
curl -O <url>                     # Сохранить файл
curl -I <url>                     # Только заголовки
wget <url>                        # Загрузить файл
wget -c <url>                     # Продолжить загрузку
ssh user@host                     # SSH подключение
scp <file> user@host:/path        # Копировать файл на удаленный хост
scp user@host:/path/file .        # Копировать файл с удаленного хоста
rsync -av <src> <dst>             # Синхронизировать директории
netstat -tuln                     # Список открытых портов
ss -tuln                          # Современная альтернатива netstat
lsof -i :8080                     # Что использует порт 8080
ifconfig                          # Сетевые интерфейсы (старый)
ip addr                           # Сетевые интерфейсы (новый)
ip route                          # Таблица маршрутизации
nslookup <domain>                 # DNS lookup
dig <domain>                      # Подробный DNS lookup
traceroute <host>                 # Трассировка маршрута
```

## Bash - Архивы и сжатие

### Работа с архивами
```bash
tar -cvf archive.tar <dir>        # Создать tar архив
tar -czvf archive.tar.gz <dir>    # Создать gzip архив
tar -cjvf archive.tar.bz2 <dir>   # Создать bzip2 архив
tar -xvf archive.tar              # Распаковать tar
tar -xzvf archive.tar.gz          # Распаковать gzip
tar -xjvf archive.tar.bz2         # Распаковать bzip2
tar -tvf archive.tar              # Просмотр содержимого
zip -r archive.zip <dir>          # Создать zip архив
unzip archive.zip                 # Распаковать zip
unzip -l archive.zip              # Просмотр содержимого zip
gzip <file>                       # Сжать файл
gunzip <file.gz>                  # Распаковать gzip
bzip2 <file>                      # Сжать bzip2
bunzip2 <file.bz2>                # Распаковать bzip2
```

## Bash - Права доступа

### Управление правами
```bash
chmod 755 <file>                  # Изменить права (rwxr-xr-x)
chmod +x <file>                   # Добавить право на выполнение
chmod -w <file>                   # Убрать право на запись
chmod u+x <file>                  # Добавить execute для owner
chmod g-w <file>                  # Убрать write для group
chmod o-r <file>                  # Убрать read для others
chmod -R 755 <dir>                # Рекурсивно изменить права
chown user <file>                 # Изменить владельца
chown user:group <file>           # Изменить владельца и группу
chown -R user:group <dir>         # Рекурсивно изменить владельца
chgrp group <file>                # Изменить группу
```

## Bash - Переменные окружения

### Работа с переменными
```bash
echo $VAR                         # Вывести переменную
export VAR="value"                # Установить переменную окружения
unset VAR                         # Удалить переменную
env                               # Список всех переменных окружения
printenv                          # Вывести переменные окружения
export PATH=$PATH:/new/path       # Добавить в PATH
echo $HOME                        # Домашняя директория
echo $USER                        # Текущий пользователь
echo $SHELL                       # Текущая оболочка
echo $PWD                         # Текущая директория
```

## Bash - История команд

### История
```bash
history                           # Показать историю команд
history | grep "pattern"          # Поиск в истории
!n                                # Выполнить команду номер n
!!                                # Повторить последнюю команду
!-2                               # Выполнить предпоследнюю
!string                           # Последняя команда начинающаяся с string
^old^new                          # Заменить old на new в последней команде
Ctrl+R                            # Интерактивный поиск в истории
history -c                        # Очистить историю
```

## Bash - Алиасы и функции

### Создание алиасов
```bash
alias ll='ls -la'                 # Создать алиас
alias gs='git status'             # Git алиас
alias ..='cd ..'                  # Навигация
alias ...='cd ../..'              # Навигация
alias grep='grep --color=auto'    # Цветной grep
unalias ll                        # Удалить алиас
alias                             # Показать все алиасы
```

### Простые функции
```bash
# Создать директорию и перейти в неё
mkcd() { mkdir -p "$1" && cd "$1"; }

# Извлечь любой архив
extract() {
    if [ -f "$1" ]; then
        case "$1" in
            *.tar.gz)  tar xvzf "$1"   ;;
            *.tar.bz2) tar xvjf "$1"   ;;
            *.zip)     unzip "$1"      ;;
            *.rar)     unrar x "$1"    ;;
            *)         echo "Неизвестный формат" ;;
        esac
    fi
}

# Быстрый backup файла
backup() { cp "$1"{,.backup-$(date +%Y%m%d-%H%M%S)}; }
```

## Bash - Горячие клавиши

### Редактирование командной строки
```
Ctrl+A          # Переход в начало строки
Ctrl+E          # Переход в конец строки
Ctrl+U          # Удалить от курсора до начала
Ctrl+K          # Удалить от курсора до конца
Ctrl+W          # Удалить слово перед курсором
Ctrl+Y          # Вставить последнее удалённое
Alt+B           # Назад на слово
Alt+F           # Вперёд на слово
Ctrl+L          # Очистить экран
Ctrl+C          # Прервать команду
Ctrl+D          # Выход (или EOF)
Ctrl+Z          # Приостановить процесс
Ctrl+R          # Поиск в истории
Tab             # Автодополнение
Tab Tab         # Показать варианты
```

## Bash - Полезные комбинации

### Часто используемые паттерны
```bash
# Найти и удалить файлы
find . -name "*.tmp" -type f -delete

# Найти большие файлы
find . -type f -size +100M -exec ls -lh {} \;

# Подсчитать файлы в директории
find . -type f | wc -l

# Рекурсивно изменить права для директорий
find . -type d -exec chmod 755 {} \;

# Рекурсивно изменить права для файлов
find . -type f -exec chmod 644 {} \;

# Найти и заменить в файлах
find . -type f -name "*.txt" -exec sed -i 's/old/new/g' {} \;

# Топ 10 самых больших файлов
du -ah . | sort -rh | head -n 10

# Топ 10 команд из истории
history | awk '{print $2}' | sort | uniq -c | sort -rn | head -10

# Мониторинг лог-файла в реальном времени
tail -f /var/log/syslog | grep "pattern"

# Очистить кэш DNS (Linux)
sudo systemd-resolve --flush-caches

# Показать открытые файлы процесса
lsof -p <pid>

# Узнать какой процесс использует файл
lsof <file>

# Рекурсивный grep с контекстом
grep -r -C 3 "pattern" .

# Создать структуру директорий
mkdir -p project/{src,tests,docs,config}

# Batch rename
for f in *.txt; do mv "$f" "${f%.txt}.md"; done

# Синхронизация с удалением
rsync -av --delete source/ destination/

# Benchmark команды
time command

# Выполнить команду несколько раз
for i in {1..5}; do echo "Iteration $i"; done

# Параллельное выполнение
command1 & command2 & wait

# Условное выполнение
command1 && command2     # command2 если command1 успешна
command1 || command2     # command2 если command1 неуспешна
```

## Дополнительные полезные команды

### Системные утилиты
```bash
clear                             # Очистить терминал
reset                             # Сбросить терминал
exit                              # Выйти из shell
logout                            # Выйти из системы
sudo command                      # Выполнить от root
sudo su                           # Переключиться на root
su - user                         # Переключиться на пользователя
passwd                            # Сменить пароль
man command                       # Руководство по команде
info command                      # Информация о команде
command --help                    # Справка по команде
apropos keyword                   # Поиск команд по ключевому слову
whatis command                    # Краткое описание команды
type command                      # Тип команды (alias/function/builtin)
```

### Мониторинг и диагностика
```bash
vmstat                            # Статистика виртуальной памяти
iostat                            # Статистика I/O
iotop                             # Мониторинг I/O процессов
dmesg                             # Сообщения ядра
journalctl                        # Системные логи (systemd)
last                              # История входов в систему
lastlog                           # Последний вход пользователей
```

---

**Совет**: Добавьте часто используемые команды и алиасы в `~/.bashrc` или `~/.bash_profile` для постоянного использования.

**Для изучения**: Команды `man`, `--help`, и `tldr` (если установлен) - ваши лучшие друзья для изучения новых команд.