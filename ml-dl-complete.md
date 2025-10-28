# Полная шпаргалка по машинному и глубокому обучению: SciPy, Scikit-learn, PyTorch

## ЧАСТЬ 1: ТЕОРИЯ И КОНЦЕПЦИИ

## Машинное обучение — основные понятия

### Типы задач
**Supervised Learning (Обучение с учителем)**
- Regression (Регрессия) — предсказание непрерывных значений
- Classification (Классификация) — предсказание категорий
- Требует размеченные данные (X, y)

**Unsupervised Learning (Обучение без учителя)**
- Clustering (Кластеризация) — группировка похожих объектов
- Dimensionality Reduction (Снижение размерности)
- Нет целевой переменной y

**Semi-supervised Learning**
- Комбинация размеченных и неразмеченных данных

**Reinforcement Learning**
- Агент учится через взаимодействие с окружением

### Основные метрики

**Классификация:**
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP) — точность положительного класса
- Recall = TP / (TP + FN) — полнота (чувствительность)
- F1-score = 2 * (Precision * Recall) / (Precision + Recall)
- ROC-AUC — площадь под кривой ROC
- Confusion Matrix — матрица ошибок

**Регрессия:**
- MSE (Mean Squared Error) = (1/n) * Σ(yi - ŷi)²
- RMSE = √MSE
- MAE (Mean Absolute Error) = (1/n) * Σ|yi - ŷi|
- R² (коэффициент детерминации) — доля объяснённой дисперсии

### Переобучение и недообучение

**Переобучение (Overfitting)**
- Модель слишком сложна для данных
- Высокая точность на тренировочных данных, низкая на тестовых
- Решение: регуляризация, больше данных, упрощение модели

**Недообучение (Underfitting)**
- Модель слишком простая
- Низкая точность везде
- Решение: усложнить модель, больше features

### Нормализация и масштабирование

**StandardScaler** — (x - mean) / std
**MinMaxScaler** — (x - min) / (max - min) [0, 1]
**RobustScaler** — устойчив к выбросам
**Normalization** — приведение к единичной норме

### Разбиение данных

- **Train/Test Split** — обычно 70-80% / 20-30%
- **K-Fold Cross-Validation** — разбиение на k фолдов
- **Stratified Split** — с сохранением пропорций классов

### Гиперпараметры

- Параметры модели, устанавливаемые перед обучением
- Grid Search — перебор всех комбинаций
- Random Search — случайный поиск
- Bayesian Optimization — оптимизация Байеса

---

## ЧАСТЬ 2: SCIPY

## SciPy — Основы

```python
import scipy
from scipy import stats, optimize, interpolate, signal, ndimage
```

### scipy.stats — Статистика

#### Распределения вероятностей
```python
from scipy.stats import norm, binom, poisson, uniform

# Нормальное распределение
norm.pdf(x)                    # Функция плотности вероятности (PDF)
norm.cdf(x)                    # Кумулятивная функция распределения (CDF)
norm.ppf(p)                    # Обратная CDF (квантиль)
norm.rvs(size=100)             # Случайные выборки

# Различные распределения
binom.pmf(k, n, p)             # Биномиальное
poisson.pmf(k, lambda)         # Пуассона
uniform.pdf(x, loc, scale)     # Равномерное
```

#### Проверка гипотез
```python
from scipy.stats import ttest_ind, ttest_rel, chi2_contingency, mannwhitneyu

# t-тест независимых выборок
stat, p_value = ttest_ind(group1, group2)

# Парный t-тест
stat, p_value = ttest_rel(before, after)

# Chi-square тест
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Mann-Whitney U test (непараметрический)
stat, p_value = mannwhitneyu(group1, group2)
```

#### Корреляция
```python
from scipy.stats import pearsonr, spearmanr

# Pearson корреляция (параметрическая)
corr, p_value = pearsonr(x, y)

# Spearman корреляция (непараметрическая)
corr, p_value = spearmanr(x, y)
```

### scipy.optimize — Оптимизация

```python
from scipy.optimize import minimize, fmin, curve_fit

# Минимизация функции
result = minimize(lambda x: x**2 + 2*x + 1, x0=0)
result.x                       # Оптимальное значение
result.fun                     # Значение функции в оптимуме

# Простой метод (Nelder-Mead)
xopt = fmin(func, x0)

# Подгонка кривой
def f(x, a, b):
    return a*x + b
popt, pcov = curve_fit(f, x_data, y_data)
```

### scipy.interpolate — Интерполяция

```python
from scipy.interpolate import interp1d, UnivariateSpline

# Линейная интерполяция
f = interp1d(x, y, kind='linear')
y_new = f(x_new)

# Сплайн интерполяция
f = UnivariateSpline(x, y, s=0)
y_new = f(x_new)
```

### scipy.signal — Обработка сигналов

```python
from scipy.signal import convolve, correlate, butter, filtfilt

# Свёртка
result = convolve(signal1, signal2)

# Фильтр Баттерворта
b, a = butter(order, critical_freq)
filtered = filtfilt(b, a, signal)
```

### scipy.ndimage — Обработка изображений

```python
from scipy.ndimage import gaussian_filter, label, find_objects

# Гауссов фильтр
smoothed = gaussian_filter(image, sigma=1)

# Маркирование компонент
labeled_array, num_features = label(image)
```

### scipy.linalg — Линейная алгебра

```python
from scipy.linalg import inv, det, eig, svd, solve

# Матричные операции
A_inv = inv(A)                 # Обратная матрица
det_A = det(A)                 # Определитель
eigenvalues, eigenvectors = eig(A)  # Собственные значения/векторы
U, s, Vt = svd(A)              # Сингулярное разложение
x = solve(A, b)                # Решение Ax = b
```

---

## ЧАСТЬ 3: SCIKIT-LEARN

## Scikit-learn — основы и установка

```python
import sklearn
from sklearn import datasets, model_selection, preprocessing, metrics
from sklearn.pipeline import Pipeline
```

## Классификация

### Логистическая регрессия
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, C=1.0, penalty='l2')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)       # Вероятности классов
```

### Метод опорных векторов (SVM)
```python
from sklearn.svm import SVC, SVR

# Классификация
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Регрессия
model = SVR(kernel='rbf', C=100, epsilon=0.1)
```

### Деревья решений
```python
from sklearn.tree import DecisionTreeClassifier, export_text

model = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Важность признаков
feature_importance = model.feature_importances_

# Визуализация дерева
tree_text = export_text(model, feature_names=feature_names)
```

### Ансамбли

#### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
feature_importance = model.feature_importances_
```

#### Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

#### AdaBoost
```python
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
model.fit(X_train, y_train)
```

### Метод k ближайших соседей (KNN)
```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Наивный Байес
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# Гауссов (для непрерывных признаков)
model = GaussianNB()

# Мультиномиальный (для дискретных, text classification)
model = MultinomialNB()
```

## Регрессия

### Линейная регрессия
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Обычная
model = LinearRegression()

# Ridge регрессия (L2 регуляризация)
model = Ridge(alpha=1.0)

# Lasso регрессия (L1 регуляризация)
model = Lasso(alpha=0.1)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model.coef_                     # Коэффициенты
model.intercept_                # Свободный член
```

### Polynomial Regression
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
```

## Кластеризация

### K-Means
```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X)
labels = model.labels_
centers = model.cluster_centers_
inertia = model.inertia_

# Предсказание для новых данных
new_labels = model.predict(X_new)
```

### DBSCAN
```python
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X)
```

### Иерархическая кластеризация
```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X)
```

### Gaussian Mixture Model
```python
from sklearn.mixture import GaussianMixture

model = GaussianMixture(n_components=3, random_state=42)
labels = model.fit_predict(X)
probabilities = model.predict_proba(X)
```

## Снижение размерности

### PCA (Principal Component Analysis)
```python
from sklearn.decomposition import PCA

model = PCA(n_components=2)
X_reduced = model.fit_transform(X)
explained_variance = model.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
```

### t-SNE
```python
from sklearn.manifold import TSNE

model = TSNE(n_components=2, perplexity=30, random_state=42)
X_reduced = model.fit_transform(X)
```

### ICA (Independent Component Analysis)
```python
from sklearn.decomposition import FastICA

model = FastICA(n_components=2, random_state=42)
X_reduced = model.fit_transform(X)
```

## Предобработка данных

### Масштабирование
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# RobustScaler (устойчив к выбросам)
scaler = RobustScaler()
```

### Кодирование категориальных переменных
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# LabelEncoder — превращает в числа
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# OneHotEncoder — создаёт бинарные признаки
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X_cat)
```

### Отбор признаков
```python
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold

# SelectKBest
selector = SelectKBest(score_func=chi2, k=10)
X_new = selector.fit_transform(X, y)

# Рекурсивное исключение (RFE)
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
X_new = rfe.fit_transform(X, y)

# Удаление низкой дисперсии
selector = VarianceThreshold(threshold=0.1)
X_new = selector.fit_transform(X)
```

## Оценка моделей

### Метрики классификации
```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Для бинарной классификации
roc_auc = roc_auc_score(y_test, y_proba[:, 1])
fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)

# Полный отчёт
print(classification_report(y_test, y_pred))
```

### Метрики регрессии
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

### Кросс-валидация
```python
from sklearn.model_selection import cross_val_score, cross_validate, KFold

# Простая кросс-валидация
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean CV score: {scores.mean():.3f} (+/- {scores.std():.3f})")

# С несколькими метриками
scoring = {'accuracy': 'accuracy', 'f1': 'f1'}
results = cross_validate(model, X, y, cv=5, scoring=scoring)
```

## Pipeline и композиция

### Pipeline
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### ColumnTransformer
```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

## Поиск гиперпараметров

### GridSearchCV
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
best_model = grid_search.best_estimator_
```

### RandomizedSearchCV
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    'C': uniform(0.1, 10),
    'gamma': ['scale', 'auto']
}

random_search = RandomizedSearchCV(SVC(), param_dist, n_iter=10, cv=5)
random_search.fit(X_train, y_train)
```

---

## ЧАСТЬ 4: PYTORCH

## PyTorch — основы

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
```

### Создание тензоров
```python
# Различные способы создания
t1 = torch.tensor([1, 2, 3])              # Из списка
t2 = torch.zeros(3, 4)                    # Нули
t3 = torch.ones(2, 5)                     # Единицы
t4 = torch.randn(3, 3)                    # Нормальное распределение
t5 = torch.randint(0, 10, (2, 3))         # Случайные целые
t6 = torch.linspace(0, 1, 5)              # Линейное пространство
t7 = torch.arange(0, 10, 2)               # С шагом
t8 = torch.eye(3)                         # Единичная матрица
```

### Операции с тензорами
```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

a + b, a - b, a * b                       # Поэлементные операции
a @ b                                     # Скалярное произведение
torch.matmul(A, B)                        # Матричное умножение

a.shape, a.dtype, a.device
a.t()                                     # Транспонирование
a.reshape(3, 1), a.view(3, 1)             # Изменение формы
a.squeeze(), a.unsqueeze(0)               # Удаление/добавление измерения
torch.cat([a, b], dim=0)                  # Объединение по оси
torch.stack([a, b], dim=0)                # Объединение новой осью
```

### Градиенты и автодифференцирование
```python
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 2*x + 1
y.backward()                              # Вычисление градиента
print(x.grad)                             # Градиент dy/dx = 2x + 2 = 6

# Для оптимизации
optimizer.zero_grad()                     # Обнуление градиентов
loss.backward()                           # Обратное распространение
optimizer.step()                          # Обновление параметров
```

### GPU/CPU
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
x = x.cuda()                              # На GPU
x = x.cpu()                               # На CPU
```

## Нейронные сети — основные компоненты

### Полносвязные слои (Dense/Linear)
```python
layer = nn.Linear(in_features=10, out_features=5)
output = layer(input_tensor)
```

### Функции активации
```python
# ReLU
relu = nn.ReLU()

# Sigmoid
sigmoid = nn.Sigmoid()

# Tanh
tanh = nn.Tanh()

# Softmax (для классификации)
softmax = nn.Softmax(dim=1)

# Leaky ReLU
leaky_relu = nn.LeakyReLU(negative_slope=0.01)

# GELU (современное)
gelu = nn.GELU()
```

### Dropout и нормализация
```python
# Dropout (для регуляризации)
dropout = nn.Dropout(p=0.5)

# Batch Normalization
bn = nn.BatchNorm1d(num_features=64)

# Layer Normalization
ln = nn.LayerNorm(normalized_shape=64)
```

### Функции потерь
```python
# Классификация
ce_loss = nn.CrossEntropyLoss()           # Кросс-энтропия

# Бинарная классификация
bce_loss = nn.BCELoss()                   # Binary cross-entropy
bce_logits_loss = nn.BCEWithLogitsLoss()  # BCE с логитами

# Регрессия
mse_loss = nn.MSELoss()                   # Mean squared error
l1_loss = nn.L1Loss()                     # Mean absolute error

# Использование
loss_value = criterion(output, target)
```

## Построение моделей

### Простая модель-класс
```python
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)          # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNet()
```

### Более сложная модель с dropout и batch norm
```python
class DenseNet(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10, dropout=0.5):
        super(DenseNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x
```

### Предварительно обученные модели (Transfer Learning)
```python
import torchvision.models as models

# ResNet18
model = models.resnet18(pretrained=True)  # pretrained=False для чистой

# VGG16
model = models.vgg16(pretrained=True)

# Изменение последнего слоя
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Заморозка параметров (кроме новых)
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True
```

## Тренировка и оптимизация

### Оптимизаторы
```python
# SGD (Стохастический градиентный спуск)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (адаптивный момент)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW (Adam с weight decay)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.01)
```

### Функция обучения (training loop)
```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy
```

### Функция валидации
```python
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy
```

### Полный цикл обучения
```python
num_epochs = 50
best_val_loss = float('inf')

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
    
    scheduler.step()  # Если используется
```

### Scheduler (расписание обучения)
```python
# Уменьшение learning rate со временем
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
```

## Работа с данными

### DataLoader
```python
# Создание датасета
dataset = TensorDataset(X_tensor, y_tensor)

# DataLoader с батчинг
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
```

### Custom Dataset
```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

dataset = CustomDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Сохранение и загрузка моделей

```python
# Сохранение весов
torch.save(model.state_dict(), 'model_weights.pth')

# Загрузка весов
model.load_state_dict(torch.load('model_weights.pth'))

# Сохранение полной модели (редко)
torch.save(model, 'full_model.pth')
model = torch.load('full_model.pth')

# Checkpoint (с оптимизатором и epoch)
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}
torch.save(checkpoint, 'checkpoint.pth')

# Загрузка checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## Популярные архитектуры

### Сверточные нейронные сети (CNN)
```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### Рекуррентные нейронные сети (RNN/LSTM)
```python
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Используем последний выход
        return out

# GRU альтернатива
self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
```

### Attention и Transformer
```python
attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

# Трансформер блок
transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer = nn.TransformerEncoder(transformer_layer, num_layers=6)
```

## Полезные утилиты

### Просмотр параметров модели
```python
# Количество параметров
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total: {total_params}, Trainable: {trainable_params}")

# Структура модели
print(model)

# Используя torchsummary (нужно установить)
from torchsummary import summary
summary(model, input_size=(3, 224, 224))
```

### Визуализация графика обучения
```python
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.show()
```

## Полезные библиотеки вместе с PyTorch

```python
# Трансформеры (BERT, GPT и т.д.)
from transformers import AutoModel, AutoTokenizer

# Компьютерное зрение
import torchvision
from torchvision import transforms, datasets, models

# Обработка текста
import torchtext

# Логирование экспериментов
import wandb
from torch.utils.tensorboard import SummaryWriter

# Облегчение тренировки
import pytorch_lightning as pl
```

---

## ЧАСТЬ 5: ПОЛЕЗНЫЕ ПАТТЕРНЫ И ЛУЧШИЕ ПРАКТИКИ

### Обработка и подготовка данных
```python
# Нормализация перед обучением
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

# Разделение на классы
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size])
```

### Регуляризация
```python
# L1 и L2 регуляризация в optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Dropout
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(50, 10)
)

# Early stopping
patience = 10
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    val_loss = validate()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        break
```

### Советы по оптимизации производительности
```python
# Использование смешанной точности (mixed precision)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for inputs, labels in train_loader:
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Параллелизм
model = nn.DataParallel(model)              # Multi-GPU на одной машине
model = model.to('cuda')

# Distributed Data Parallel
from torch.nn.parallel import DistributedDataParallel
model = DistributedDataParallel(model)
```

---

**Документация:**
- SciPy: https://docs.scipy.org/
- Scikit-learn: https://scikit-learn.org/
- PyTorch: https://pytorch.org/docs/stable/
- Scikit-learn метрики: https://scikit-learn.org/stable/modules/model_evaluation.html

**Практика и ресурсы:**
- Kaggle — датасеты и соревнования
- Papers with Code — реализации научных работ
- Google Colab — бесплатный GPU для экспериментов
- Awesome-Machine-Learning на GitHub