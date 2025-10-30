# Шпаргалка по XML моделям MuJoCo

Полное руководство по созданию моделей роботов в MuJoCo через XML.

---

## Базовая структура XML файла

```xml
<mujoco model="название_модели">
  <option/>           <!-- Глобальные настройки симуляции -->
  <asset/>            <!-- Ресурсы: текстуры, меши -->
  <worldbody/>        <!-- Иерархия тел и геометрии -->
  <actuator/>         <!-- Приводы (моторы, цилиндры) -->
  <sensor/>           <!-- Датчики -->
  <contact/>          <!-- Настройки контактов -->
</mujoco>
```

---

## 1. Корневой элемент `<mujoco>`

```xml
<mujoco model="my_robot">
  <!-- Вся модель здесь -->
</mujoco>
```

**Атрибуты:**
- `model="название"` — имя модели (опционально)

---

## 2. Глобальные настройки `<option>`

```xml
<option timestep="0.001"           <!-- Шаг симуляции (сек) -->
        gravity="0 0 -9.81"        <!-- Вектор гравитации (м/с²) -->
        integrator="Euler"         <!-- Интегратор: Euler, RK4, implicit -->
        collision="all"            <!-- Проверка коллизий: all, predefined, dynamic -->
        cone="pyramidal"           <!-- Модель трения: pyramidal, elliptic -->
        iterations="50"            <!-- Итерации решателя -->
        solver="Newton"/>          <!-- Решатель: Newton, CG, PGS -->
```

**Типичные значения:**
- `timestep="0.001"` до `0.002` — баланс точности и скорости
- `gravity="0 0 -9.81"` — стандартная земная гравитация
- `integrator="Euler"` — самый быстрый

---

## 3. Визуальные настройки `<visual>`

```xml
<visual>
  <global offwidth="1920"          <!-- Ширина рендера -->
          offheight="1080"         <!-- Высота рендера -->
          fovy="45"/>              <!-- Поле зрения камеры -->
  <quality shadowsize="4096"/>     <!-- Качество теней -->
  <map fogstart="3"                <!-- Начало тумана -->
       fogend="10"/>               <!-- Конец тумана -->
</visual>
```

---

## 4. Ресурсы `<asset>`

### 4.1 Текстуры

```xml
<asset>
  <texture name="grid" type="2d" 
           builtin="checker" 
           rgb1="0.2 0.3 0.4" 
           rgb2="0.3 0.4 0.5" 
           width="512" height="512"/>
  
  <texture name="skybox" type="skybox" 
           builtin="gradient" 
           rgb1="0.3 0.5 0.7" 
           rgb2="0 0 0" 
           width="512" height="512"/>
</asset>
```

### 4.2 Материалы

```xml
<asset>
  <material name="mat_metal" 
            texture="grid" 
            texrepeat="1 1" 
            specular="1" 
            shininess="0.3" 
            reflectance="0.5"/>
</asset>
```

### 4.3 Внешние меши

```xml
<asset>
  <mesh name="arm_mesh" file="arm.stl" scale="0.001 0.001 0.001"/>
</asset>
```

---

## 5. Мировое пространство `<worldbody>`

Содержит всю иерархию тел, геометрии и источников света.

### 5.1 Базовая геометрия (пол)

```xml
<worldbody>
  <!-- Пол -->
  <geom name="floor" 
        type="plane" 
        size="10 10 0.1" 
        pos="0 0 0" 
        rgba="0.8 0.8 0.8 1"/>
  
  <!-- Освещение -->
  <light diffuse="1 1 1" 
         pos="0 0 3" 
         dir="0 0 -1"/>
</worldbody>
```

### 5.2 Тела `<body>`

```xml
<body name="link1" pos="0 0 0.5">
  <!-- Визуальная геометрия -->
  <geom name="visual_link1" 
        type="cylinder" 
        fromto="0 0 0 0 0 0.5" 
        size="0.05" 
        rgba="0.2 0.6 0.8 1"/>
  
  <!-- Вложенное тело -->
  <body name="link2" pos="0 0 0.5">
    <!-- ... -->
  </body>
</body>
```

---

## 6. Типы геометрии `<geom>`

### 6.1 Плоскость (plane)

```xml
<geom type="plane" size="10 10 0.1" pos="0 0 0"/>
```
- `size`: [полуширина_X, полуширина_Y, не_используется]

### 6.2 Сфера (sphere)

```xml
<geom type="sphere" size="0.1" pos="0 0 1"/>
```
- `size`: радиус

### 6.3 Капсула (capsule)

```xml
<geom type="capsule" fromto="0 0 0 1 0 0" size="0.05"/>
```
- `fromto`: [x1 y1 z1 x2 y2 z2] — начало и конец
- `size`: радиус

### 6.4 Цилиндр (cylinder)

```xml
<geom type="cylinder" fromto="0 0 0 0 0 1" size="0.05"/>
```
- `fromto`: координаты начала и конца
- `size`: радиус

### 6.5 Параллелепипед (box)

```xml
<geom type="box" size="0.5 0.3 0.2" pos="0 0 1"/>
```
- `size`: [полуразмер_X, полуразмер_Y, полуразмер_Z]

### 6.6 Меш (mesh)

```xml
<geom type="mesh" mesh="arm_mesh" pos="0 0 0"/>
```

---

## 7. Суставы `<joint>`

### 7.1 Вращательный (hinge)

```xml
<joint name="joint1" 
       type="hinge" 
       axis="0 0 1"              <!-- Ось вращения -->
       pos="0 0 0"               <!-- Позиция шарнира -->
       range="-3.14 3.14"        <!-- Диапазон (рад) -->
       damping="0.1"             <!-- Демпфирование -->
       armature="0.01"           <!-- Инерция привода -->
       frictionloss="0.05"       <!-- Потери на трение -->
       limited="true"/>          <!-- Ограничение по range -->
```

### 7.2 Поступательный (slide)

```xml
<joint name="prismatic1" 
       type="slide" 
       axis="1 0 0"              <!-- Направление движения -->
       range="-1 1"              <!-- Диапазон (м) -->
       damping="10"/>
```

### 7.3 Свободный (free) — 6 степеней свободы

```xml
<joint name="free_joint" type="free"/>
```

### 7.4 Шаровой шарнир (ball)

```xml
<joint name="ball_joint" type="ball" damping="0.5"/>
```

---

## 8. Инерция `<inertial>`

Определяет массу и момент инерции тела.

```xml
<inertial pos="0.25 0 0"           <!-- Центр масс -->
          mass="1.0"               <!-- Масса (кг) -->
          diaginertia="0.01 0.01 0.005"/>  <!-- Главные моменты инерции -->
```

**Или задать полную матрицу:**

```xml
<inertial pos="0 0 0" 
          mass="2.0" 
          fullinertia="0.1 0.2 0.15 0.01 0.02 0.005"/>
<!-- fullinertia: [Ixx Iyy Izz Ixy Ixz Iyz] -->
```

**Автоматический расчёт из геометрии:**

```xml
<geom type="box" size="0.5 0.3 0.2" mass="1"/>
<!-- MuJoCo автоматически вычислит инерцию -->
```

---

## 9. Актюаторы `<actuator>`

### 9.1 Мотор (motor)

```xml
<actuator>
  <motor name="motor1" 
         joint="joint1"            <!-- Управляемый сустав -->
         gear="100"                <!-- Передаточное число -->
         ctrllimited="true" 
         ctrlrange="-10 10"/>      <!-- Диапазон момента (Н·м) -->
</actuator>
```

### 9.2 Позиционный сервопривод (position)

```xml
<actuator>
  <position name="servo1" 
            joint="joint1" 
            kp="100"               <!-- Коэффициент P -->
            ctrlrange="-3.14 3.14"/>
</actuator>
```

### 9.3 Скоростной привод (velocity)

```xml
<actuator>
  <velocity name="velocity_ctrl" 
            joint="joint1" 
            kv="10"                <!-- Коэффициент усиления -->
            ctrlrange="-5 5"/>     <!-- Диапазон скорости (рад/с) -->
</actuator>
```

### 9.4 Цилиндр (cylinder) — линейный актюатор

```xml
<actuator>
  <cylinder name="piston" 
            joint="prismatic1" 
            area="0.01"            <!-- Площадь поршня (м²) -->
            bias="0 0 1"           <!-- Направление силы -->
            ctrllimited="true" 
            ctrlrange="0 1000"/>   <!-- Диапазон давления (Па) -->
</actuator>
```

---

## 10. Датчики `<sensor>`

### 10.1 Датчик позиции сустава

```xml
<sensor>
  <jointpos name="sensor_pos" joint="joint1"/>
</sensor>
```

### 10.2 Датчик скорости сустава

```xml
<sensor>
  <jointvel name="sensor_vel" joint="joint1"/>
</sensor>
```

### 10.3 Акселерометр

```xml
<sensor>
  <accelerometer name="imu_acc" site="imu_site"/>
</sensor>
```

### 10.4 Гироскоп

```xml
<sensor>
  <gyro name="imu_gyro" site="imu_site"/>
</sensor>
```

### 10.5 Датчик силы/момента

```xml
<sensor>
  <force name="force_sensor" site="contact_site"/>
  <torque name="torque_sensor" site="contact_site"/>
</sensor>
```

---

## 11. Точки привязки `<site>`

Используются для датчиков, камер, маркеров.

```xml
<body name="link1">
  <site name="imu_site" 
        pos="0 0 0.25" 
        size="0.01" 
        rgba="1 0 0 1"/>
</body>
```

---

## 12. Камеры `<camera>`

```xml
<worldbody>
  <camera name="fixed" 
          pos="2 2 2" 
          xyaxes="-1 1 0 0 0 1"   <!-- Ориентация -->
          fovy="45"/>
  
  <body name="robot">
    <camera name="onboard" 
            pos="0 0 0.5" 
            euler="0 0 0"/>
  </body>
</worldbody>
```

---

## 13. Контакты и столкновения

### 13.1 Пары контактов

```xml
<contact>
  <pair geom1="floor" geom2="foot" 
        friction="1 0.005 0.0001"  <!-- [μ, torsion, rolling] -->
        solref="0.02 1"            <!-- Жёсткость/демпфирование -->
        solimp="0.9 0.95 0.001"/>  <!-- Параметры импульса -->
</contact>
```

### 13.2 Исключение контактов

```xml
<contact>
  <exclude body1="link1" body2="link2"/>
</contact>
```

---

## 14. Полный пример: Простая роборука

```xml
<mujoco model="simple_arm">
  <!-- Настройки -->
  <option timestep="0.001" gravity="0 0 -9.81"/>
  
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>
  
  <!-- Ресурсы -->
  <asset>
    <texture name="grid" type="2d" builtin="checker" 
             rgb1="0.2 0.3 0.4" rgb2="0.3 0.4 0.5" 
             width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="1 1"/>
  </asset>
  
  <!-- Мир -->
  <worldbody>
    <!-- Пол -->
    <geom name="floor" type="plane" size="5 5 0.1" 
          material="grid_mat"/>
    
    <!-- Освещение -->
    <light diffuse="1 1 1" pos="0 0 3" dir="0 0 -1"/>
    
    <!-- База -->
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.1 0.05" rgba="0.3 0.3 0.3 1"/>
      <inertial pos="0 0 0" mass="5" diaginertia="0.1 0.1 0.1"/>
      
      <!-- Звено 1 -->
      <body name="link1" pos="0 0 0.05">
        <joint name="joint1" type="hinge" axis="0 0 1" 
               range="-3.14 3.14" damping="0.1"/>
        
        <geom name="link1_geom" type="capsule" 
              fromto="0 0 0 0.5 0 0" size="0.05" 
              rgba="0.2 0.6 0.8 1"/>
        
        <inertial pos="0.25 0 0" mass="1" 
                  diaginertia="0.01 0.01 0.005"/>
        
        <!-- Звено 2 -->
        <body name="link2" pos="0.5 0 0">
          <joint name="joint2" type="hinge" axis="0 0 1" 
                 range="-3.14 3.14" damping="0.1"/>
          
          <geom name="link2_geom" type="capsule" 
                fromto="0 0 0 0.3 0 0" size="0.04" 
                rgba="0.8 0.2 0.2 1"/>
          
          <inertial pos="0.15 0 0" mass="0.5" 
                    diaginertia="0.005 0.005 0.002"/>
          
          <!-- Схват -->
          <body name="gripper" pos="0.3 0 0">
            <geom type="sphere" size="0.05" rgba="1 1 0 1"/>
            <site name="ee_site" pos="0 0 0" size="0.01"/>
          </body>
        </body>
      </body>
    </body>
    
    <!-- Камера -->
    <camera name="track" mode="trackcom" pos="0 -2 1" 
            xyaxes="1 0 0 0 0.5 1"/>
  </worldbody>
  
  <!-- Актюаторы -->
  <actuator>
    <motor name="motor1" joint="joint1" gear="50" 
           ctrllimited="true" ctrlrange="-20 20"/>
    <motor name="motor2" joint="joint2" gear="30" 
           ctrllimited="true" ctrlrange="-10 10"/>
  </actuator>
  
  <!-- Датчики -->
  <sensor>
    <jointpos name="pos1" joint="joint1"/>
    <jointpos name="pos2" joint="joint2"/>
    <jointvel name="vel1" joint="joint1"/>
    <jointvel name="vel2" joint="joint2"/>
  </sensor>
</mujoco>
```

---

## 15. Полезные атрибуты

### Общие для `<geom>`

```xml
<geom name="..."           <!-- Уникальное имя -->
      type="..."           <!-- Тип: plane, sphere, capsule, cylinder, box, mesh -->
      pos="x y z"          <!-- Позиция относительно родителя -->
      quat="w x y z"       <!-- Кватернион ориентации -->
      euler="x y z"        <!-- Углы Эйлера (XYZ, градусы) -->
      rgba="r g b a"       <!-- Цвет (0-1) -->
      material="..."       <!-- Ссылка на материал -->
      mass="..."           <!-- Масса (альтернатива inertial) -->
      density="..."        <!-- Плотность (кг/м³) -->
      friction="..."       <!-- Коэффициент трения [μ, torsion, rolling] -->
      contype="..."        <!-- Битовая маска типа контакта -->
      conaffinity="..."    <!-- Битовая маска аффинности контакта -->
      condim="..."         <!-- Размерность контакта (1-6) -->
      group="..."          <!-- Группа визуализации (0-5) -->
/>
```

### Общие для `<body>`

```xml
<body name="..."           <!-- Уникальное имя -->
      pos="x y z"          <!-- Позиция -->
      quat="w x y z"       <!-- Ориентация (кватернион) -->
      euler="x y z"        <!-- Ориентация (углы Эйлера) -->
      childclass="..."     <!-- Класс по умолчанию для детей -->
/>
```

---

## 16. Система классов (для переиспользования)

```xml
<default>
  <default class="arm_link">
    <geom type="capsule" rgba="0.2 0.6 0.8 1" size="0.05"/>
    <joint type="hinge" damping="0.1" armature="0.01"/>
  </default>
</default>

<worldbody>
  <body name="link1" childclass="arm_link">
    <joint name="j1" axis="0 0 1"/>
    <geom fromto="0 0 0 1 0 0"/>
  </body>
</worldbody>
```

---

## 17. Советы и лучшие практики

### ✅ DO (Делать)

1. **Используйте осмысленные имена** — `shoulder_joint`, а не `j1`
2. **Начинайте с простых геометрий** — капсулы вместо мешей
3. **Проверяйте массы** — реалистичные значения (кг)
4. **Добавляйте демпфирование** — `damping="0.1"` для стабильности
5. **Используйте `<site>` для маркеров** — удобно для визуализации
6. **Группируйте по классам** — переиспользование параметров

### ❌ DON'T (Не делать)

1. **Не делайте timestep < 0.0005** — слишком медленно
2. **Не забывайте про инерцию** — без неё тела невесомы
3. **Не создавайте геометрии без коллизий там, где нужны контакты**
4. **Не делайте слишком маленькие геометрии** — < 0.001 м проблематично
5. **Не игнорируйте ограничения суставов** — `range` и `limited="true"`

---

## 18. Отладка модели

### Загрузка и проверка в Python

```python
import mujoco

try:
    model = mujoco.MjModel.from_xml_path('robot.xml')
    print("✓ Модель загружена успешно")
    print(f"  Степеней свободы: {model.nq}")
    print(f"  Актюаторов: {model.nu}")
except Exception as e:
    print(f"✗ Ошибка: {e}")
```

### Визуализация в viewer

```bash
python -m mujoco.viewer --mjcf=robot.xml
```

---

## 19. Полезные ссылки

- **Официальная документация:** https://mujoco.readthedocs.io/
- **XML Reference:** https://mujoco.readthedocs.io/en/stable/XMLreference.html
- **Примеры моделей:** `~/.mujoco/mujoco-x.x.x/model/`
- **Форум:** https://github.com/google-deepmind/mujoco/discussions

---

## 20. Шаблон для быстрого старта

```xml
<mujoco model="template">
  <option timestep="0.001"/>
  
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1"/>
    <light diffuse="1 1 1" pos="0 0 3" dir="0 0 -1"/>
    
    <body name="robot" pos="0 0 0.5">
      <joint name="joint1" type="hinge" axis="0 0 1"/>
      <geom name="link1" type="capsule" fromto="0 0 0 1 0 0" size="0.05"/>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="motor1" joint="joint1" ctrllimited="true" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
```

---

**Готово!** Эта шпаргалка покрывает 95% использования MuJoCo XML. 🎯
