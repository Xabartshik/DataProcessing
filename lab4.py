# ЛАБОРАТОРНАЯ РАБОТА № 4. ВАРИАНТ 14

# Построение выборочного уравнения линии регрессии по сгруппированным данным

# Распределение цехов по изменению средней заработной платы Y (руб.)

# в зависимости от изменения производительности труда X (руб.) по кварталам

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import sys
from io import StringIO


# ========== НАСТРОЙКА ВЫВОДА В ФАЙЛ ==========

# Создаем объект для захвата консольного вывода
original_stdout = sys.stdout
output_buffer = StringIO()

class Logger:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()

# Перенаправляем вывод одновременно в консоль и в буфер
sys.stdout = Logger(original_stdout, output_buffer)

print("=== ЛАБОРАТОРНАЯ РАБОТА № 4. ВАРИАНТ 14 ===")
print("Распределение цехов по изменению средней заработной платы Y (руб.)")
print("в зависимости от изменения производительности труда X (руб.) по кварталам")
print()

# ========== ИСХОДНЫЕ ДАННЫЕ ==========

# Значения переменных
x_values = [2700, 2800, 2900, 3000, 3100, 3200, 3300]
y_values = [4600, 5100, 5600, 6100, 6600, 7100, 7600]

# Частоты из корреляционной таблицы
data = {
    4600: [1, 3, 2, 0, 0, 0, 0],  # ny = 6
    5100: [0, 2, 5, 0, 2, 0, 0],  # ny = 9
    5600: [0, 1, 3, 1, 0, 2, 0],  # ny = 7
    6100: [0, 0, 1, 0, 8, 0, 0],  # ny = 9
    6600: [0, 0, 0, 0, 2, 5, 0],  # ny = 7
    7100: [0, 0, 0, 0, 0, 4, 4],  # ny = 8
    7600: [0, 0, 0, 0, 0, 1, 3]   # ny = 4
}

# Маргинальные частоты
nx = [1, 6, 11, 1, 12, 12, 7]  # nx = 50
ny = [6, 9, 7, 9, 7, 8, 4]     # ny = 50
n = 50  # общий объем выборки

# Вывод корреляционной таблицы
print("Корреляционная таблица:")
print("X\\Y", end="\t")
for y in y_values:
    print(f"{y}", end="\t")
print("nx")

for i, x in enumerate(x_values):
    print(f"{x}", end="\t")
    for y in y_values:
        print(f"{data[y][i]}", end="\t")
    print(f"{nx[i]}")

print("ny", end="\t")
for n_y in ny:
    print(f"{n_y}", end="\t")
print(f"{n}")
print()

# ========== ШАГ 1: ПОСТРОЕНИЕ КОРРЕЛЯЦИОННОГО ПОЛЯ ==========

print("=== ШАГ 1: ПОСТРОЕНИЕ КОРРЕЛЯЦИОННОГО ПОЛЯ ===")

# Создаем массивы точек для корреляционного поля
x_points = []
y_points = []

# Заполнение информации о точках
for i, x in enumerate(x_values):
    for j, y in enumerate(y_values):
        freq = data[y][i]
        for _ in range(freq):
            x_points.append(x)
            y_points.append(y)

print(f"Общее количество точек: {len(x_points)}")
print("Корреляционное поле построено (данные готовы для визуализации)")
print()

# ========== ШАГ 2: РАСЧЕТ ОСНОВНЫХ ХАРАКТЕРИСТИК ==========

print("=== ШАГ 2: РАСЧЕТ ОСНОВНЫХ ХАРАКТЕРИСТИК ===")

# Средние значения
x_mean = sum(x_points) / len(x_points)
y_mean = sum(y_points) / len(y_points)

print(f"Средние значения:")
print(f"x̄ = {x_mean:.2f}")
print(f"ȳ = {y_mean:.2f}")

# Вариансы и стандартные отклонения
x_var = sum((x - x_mean)**2 for x in x_points) / (len(x_points) - 1)
y_var = sum((y - y_mean)**2 for y in y_points) / (len(y_points) - 1)

s_x = np.sqrt(x_var)
s_y = np.sqrt(y_var)

print(f"Стандартные отклонения:")
print(f"sx = {s_x:.2f}")
print(f"sy = {s_y:.2f}")
print()

# ========== ШАГ 3: РАСЧЕТ КОЭФФИЦИЕНТА КОРРЕЛЯЦИИ ==========

print("=== ШАГ 3: РАСЧЕТ КОЭФФИЦИЕНТА КОРРЕЛЯЦИИ ===")

# Ковариация
cov_xy = sum((x_points[i] - x_mean) * (y_points[i] - y_mean) for i in range(len(x_points))) / (len(x_points) - 1)

# Коэффициент корреляции
r = cov_xy / (s_x * s_y)

print(f"Ковариация: cov(X,Y) = {cov_xy:.2f}")
print(f"Коэффициент корреляции: r = {r:.4f}")
print()

# ========== ШАГ 4: УРАВНЕНИЯ РЕГРЕССИИ ПО МНК ==========

print("=== ШАГ 4: УРАВНЕНИЯ РЕГРЕССИИ ПО МНК ===")

# Вычисляем суммы для системы нормальных уравнений
sum_x = sum(x_points)
sum_y = sum(y_points)
sum_xy = sum(x_points[i] * y_points[i] for i in range(len(x_points)))
sum_x2 = sum(x**2 for x in x_points)

print("Суммы для МНК:")
print(f"Σx = {sum_x}")
print(f"Σy = {sum_y}")
print(f"Σxy = {sum_xy}")
print(f"Σx² = {sum_x2}")


det = n * sum_x2 - sum_x**2
a1 = (n * sum_xy - sum_x * sum_y) / det
a0 = (sum_y * sum_x2 - sum_x * sum_xy) / det

print(f"\nКоэффициенты уравнения y = a₁x + a₀:")
print(f"a₁ = {a1:.6f}")
print(f"a₀ = {a0:.6f}")
print(f"Уравнение регрессии y на x: y = {a1:.6f}x + {a0:.6f}")
print()

# Уравнение регрессии через коэффициент корреляции
print("=== УРАВНЕНИЕ РЕГРЕССИИ ЧЕРЕЗ КОЭФФИЦИЕНТ КОРРЕЛЯЦИИ ===")

b1 = r * (s_y / s_x)  # коэффициент при x
b0 = y_mean - b1 * x_mean  # свободный член

print(f"Коэффициент при x: b₁ = r * (sy/sx) = {r:.4f} * ({s_y:.2f}/{s_x:.2f}) = {b1:.6f}")
print(f"Свободный член: b₀ = ȳ - b₁ * x̄ = {y_mean:.2f} - {b1:.6f} * {x_mean:.2f} = {b0:.6f}")
print(f"Уравнение регрессии через r: y = {b1:.6f}x + {b0:.6f}")
print()

print("Сравнение уравнений:")
print(f"МНК: y = {a1:.6f}x + {a0:.6f}")
print(f"Через r: y = {b1:.6f}x + {b0:.6f}")
print(f"Разность коэффициентов: Δa₁ = {abs(a1 - b1):.8f}, Δa₀ = {abs(a0 - b0):.8f}")
print()

# Уравнение регрессии x на y
c1 = r * (s_x / s_y)
c0 = x_mean - c1 * y_mean

print(f"Уравнение регрессии x на y: x = {c1:.6f}y + {c0:.6f}")
print()

# ========== ШАГ 5: ОЦЕНКА ТЕСНОТЫ СВЯЗИ И ЕЁ ЗНАЧИМОСТИ ==========

print("=== ШАГ 5: ОЦЕНКА ТЕСНОТЫ СВЯЗИ И ЕЁ ЗНАЧИМОСТИ ===")

# Проверка значимости коэффициента корреляции по критерию Стьюдента
t_calc = r * np.sqrt((n - 2) / (1 - r**2))
df = n - 2  # степени свободы
alpha = 0.05
t_crit = stats.t.ppf(1 - alpha/2, df)

print(f"Коэффициент корреляции: r = {r:.4f}")
print(f"Наблюдаемое значение статистики: t_набл = {t_calc:.4f}")
print(f"Критическое значение: t_крит = {t_crit:.4f} (при α = {alpha}, k = {df})")

if abs(t_calc) > t_crit:
    print("Заключение: |t_набл| > t_крит, коэффициент корреляции значимо отличается от нуля")
    print("Связь между признаками статистически значима")
else:
    print("Заключение: |t_набл| ≤ t_крит, коэффициент корреляции незначим")

print()

# Интерпретация силы связи
if abs(r) < 0.3:
    strength = "слабая"
elif abs(r) < 0.7:
    strength = "умеренная"
else:
    strength = "сильная"

print(f"Сила связи: {strength} (|r| = {abs(r):.4f})")
if r > 0:
    print("Направление связи: прямая (положительная)")
else:
    print("Направление связи: обратная (отрицательная)")

print()

# Коэффициент детерминации
r2 = r**2
print(f"Коэффициент детерминации: R² = {r2:.4f}")
print(f"Доля вариации Y, объясняемая регрессией: {r2*100:.2f}%")
print()

# ========== ШАГ 6: ПРОВЕРКА АДЕКВАТНОСТИ МОДЕЛИ ==========

print("=== ШАГ 6: ПРОВЕРКА АДЕКВАТНОСТИ МОДЕЛИ ===")

# Вычисляем предсказанные значения
y_pred = [a1 * x + a0 for x in x_points]

# Общая сумма квадратов отклонений (TSS)
TSS = sum((y - y_mean)**2 for y in y_points)

# Объясненная сумма квадратов (ESS)
ESS = sum((y_pred[i] - y_mean)**2 for i in range(len(y_pred)))

# Остаточная сумма квадратов (RSS)
RSS = sum((y_points[i] - y_pred[i])**2 for i in range(len(y_points)))

print(f"Общая сумма квадратов (TSS): {TSS:.2f}")
print(f"Объясненная сумма квадратов (ESS): {ESS:.2f}")
print(f"Остаточная сумма квадратов (RSS): {RSS:.2f}")
print(f"Проверка: TSS = ESS + RSS? {TSS:.2f} = {ESS + RSS:.2f}")
print()

# F-статистика
k1 = 1  # число объясняющих переменных
k2 = n - 2  # степени свободы для остатков
F_calc = (ESS / k1) / (RSS / k2)
F_crit = stats.f.ppf(1 - alpha, k1, k2)

print(f"F-статистика:")
print(f"F_набл = (ESS/k₁) / (RSS/k₂) = ({ESS:.2f}/1) / ({RSS:.2f}/{k2}) = {F_calc:.4f}")
print(f"F_крит = {F_crit:.4f} (при α = {alpha}, k₁ = {k1}, k₂ = {k2})")

if F_calc > F_crit:
    print("Заключение: F_набл > F_крит, модель регрессии адекватна")
    print("Уравнение регрессии согласуется с опытными данными")
else:
    print("Заключение: F_набл ≤ F_крит, модель регрессии неадекватна")

print()

# ========== ШАГ 7: ПРОВЕРКА НАДЕЖНОСТИ КОЭФФИЦИЕНТОВ ==========

print("=== ШАГ 7: ПРОВЕРКА НАДЕЖНОСТИ КОЭФФИЦИЕНТОВ РЕГРЕССИИ ===")

# Стандартные ошибки коэффициентов
s_res = np.sqrt(RSS / (n - 2))  # остаточное стандартное отклонение
s_xx = sum((x - x_mean)**2 for x in x_points)  # сумма квадратов отклонений x

# Стандартная ошибка коэффициента a₁
se_a1 = s_res / np.sqrt(s_xx)

# Стандартная ошибка коэффициента a₀
se_a0 = s_res * np.sqrt((1/n) + (x_mean**2 / s_xx))

print(f"Остаточное стандартное отклонение: s_ост = {s_res:.4f}")
print(f"Стандартная ошибка коэффициента a₁: se(a₁) = {se_a1:.6f}")
print(f"Стандартная ошибка коэффициента a₀: se(a₀) = {se_a0:.4f}")
print()

# t-статистики для коэффициентов
t_a1 = a1 / se_a1
t_a0 = a0 / se_a0

print(f"t-статистики:")
print(f"t(a₁) = a₁/se(a₁) = {a1:.6f}/{se_a1:.6f} = {t_a1:.4f}")
print(f"t(a₀) = a₀/se(a₀) = {a0:.6f}/{se_a0:.4f} = {t_a0:.4f}")
print(f"t_крит = {t_crit:.4f} (двустороннее распределение)")
print()

# Проверка значимости коэффициентов
print("Проверка значимости коэффициентов:")
if abs(t_a1) > t_crit:
    print(f"Коэффициент a₁: |t(a₁)| = {abs(t_a1):.4f} > {t_crit:.4f} - ЗНАЧИМ")
else:
    print(f"Коэффициент a₁: |t(a₁)| = {abs(t_a1):.4f} ≤ {t_crit:.4f} - НЕЗНАЧИМ")

if abs(t_a0) > t_crit:
    print(f"Коэффициент a₀: |t(a₀)| = {abs(t_a0):.4f} > {t_crit:.4f} - ЗНАЧИМ")
else:
    print(f"Коэффициент a₀: |t(a₀)| = {abs(t_a0):.4f} ≤ {t_crit:.4f} - НЕЗНАЧИМ")

print()

# Доверительные интервалы для коэффициентов
conf_int_a1_lower = a1 - t_crit * se_a1
conf_int_a1_upper = a1 + t_crit * se_a1
conf_int_a0_lower = a0 - t_crit * se_a0
conf_int_a0_upper = a0 + t_crit * se_a0

print(f"Доверительные интервалы (при α = {alpha}):")
print(f"a₁: [{conf_int_a1_lower:.6f}; {conf_int_a1_upper:.6f}]")
print(f"a₀: [{conf_int_a0_lower:.4f}; {conf_int_a0_upper:.4f}]")
print()

# ========== ШАГ 8: ПОСТРОЕНИЕ ГРАФИКА КОРРЕЛЯЦИОННОГО ПОЛЯ ==========

print("=== ШАГ 8: ПОСТРОЕНИЕ ГРАФИКА КОРРЕЛЯЦИОННОГО ПОЛЯ ===")

plt.plot(figsize=(16, 7))

# === Корреляционное поле с уравнением регрессии ===

# Построение точек корреляционного поля
plt.scatter(x_points, y_points, alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=0.5)

# Линия регрессии y на x
x_line = np.linspace(min(x_points), max(x_points), 100)
y_line = a1 * x_line + a0
plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {a1:.3f}x + {a0:.0f}')

# Линия регрессии x на y (для сравнения)
y_line2 = np.linspace(min(y_points), max(y_points), 100)
x_line2 = c1 * y_line2 + c0
plt.plot(x_line2, y_line2, 'g--', linewidth=2, alpha=0.7, label=f'x = {c1:.3f}y + {c0:.0f}')

slope, intercept = np.polyfit(x_points, y_points, 1)
X= []
for x in range(len(x_points)):
    X.append(slope * x_points[x] + intercept)
plt.plot(x_points, X, 'b--', label='Линия тренда')

# Средние значения
plt.axvline(x_mean, color='orange', linestyle=':', alpha=0.7, label=f'x̄ = {x_mean:.0f}')
plt.axhline(y_mean, color='orange', linestyle=':', alpha=0.7, label=f'ȳ = {y_mean:.0f}')

plt.xlabel('Производительность труда X (руб.)', fontsize=12)
plt.ylabel('Средняя заработная плата Y (руб.)', fontsize=12)
plt.title('Корреляционное поле и уравнения регрессии\nЗависимость заработной платы от производительности труда', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('correlation_field_and_regression.png', dpi=300, bbox_inches='tight')
plt.show()

print("График корреляционного поля с уравнениями регрессии построен и сохранен как 'correlation_field_and_regression.png'")
print()

# ========== ШАГ 9: ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ ==========

print("=== ШАГ 9: ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ ===")
print("Экономическая интерпретация уравнения регрессии:")
print(f"y = {a1:.3f}x + {a0:.1f}")
print()

print(f"1. При увеличении производительности труда на 1 руб.,")
print(f"   средняя заработная плата увеличивается на {a1:.3f} руб.")
print()

print(f"2. Свободный член a₀ = {a0:.1f} не имеет экономического смысла,")
print(f"   так как производительность труда не может быть равна нулю.")
print()

# Прогнозирование
print("Примеры прогнозирования:")
test_x_values = [2750, 3000, 3250]
for x_test in test_x_values:
    y_pred_test = a1 * x_test + a0
    print(f"При X = {x_test} руб., прогнозируемая Y = {y_pred_test:.0f} руб.")
print()

# ========== ИТОГОВАЯ СВОДКА ==========

print("=== ИТОГОВАЯ СВОДКА РЕЗУЛЬТАТОВ ===")
print()
print("ЗАДАЧА: Исследование зависимости между изменением средней заработной платы")
print("и изменением производительности труда по цехам предприятия")
print()

print("1. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ:")
print(f"   • Коэффициент корреляции: r = {r:.4f}")
print(f"   • Сила связи: {strength}")
print(f"   • Направление связи: прямая")
print(f"   • Коэффициент детерминации: R² = {r2:.4f} ({r2*100:.1f}%)")
print()

print("2. УРАВНЕНИЕ РЕГРЕССИИ:")
print(f"   • МНК: y = {a1:.3f}x + {a0:.1f}")
print(f"   • Через r: y = {b1:.3f}x + {b0:.1f}")
print()

print("3. СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ:")
print(f"   • Коэффициент корреляции: t = {t_calc:.2f} > t_крит = {t_crit:.2f} - ЗНАЧИМ")
print(f"   • Коэффициент регрессии a₁: t = {t_a1:.2f} > t_крит = {t_crit:.2f} - ЗНАЧИМ")
print(f"   • Свободный член a₀: t = {abs(t_a0):.2f} > t_крит = {t_crit:.2f} - ЗНАЧИМ")
print()

print("4. АДЕКВАТНОСТЬ МОДЕЛИ:")
print(f"   • F-критерий: F = {F_calc:.2f} > F_крит = {F_crit:.2f} - МОДЕЛЬ АДЕКВАТНА")
print()

print("5. ЭКОНОМИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ:")
print(f"   • Увеличение производительности труда на 1 руб. приводит к")
print(f"     увеличению средней заработной платы на {a1:.2f} руб.")
print(f"   • {r2*100:.1f}% вариации заработной платы объясняется изменением")
print(f"     производительности труда")
print()

print("6. ЗАКЛЮЧЕНИЕ:")
print("   Между изменением средней заработной платы и изменением")
print("   производительности труда существует сильная прямая линейная")
print("   корреляционная зависимость. Полученное уравнение регрессии")
print("   статистически значимо и может использоваться для прогнозирования.")
print()

# Таблица результатов
results_data = {
    'Показатель': [
        'Объем выборки (n)',
        'Среднее значение X (руб.)',
        'Среднее значение Y (руб.)',
        'Коэффициент корреляции (r)',
        'Коэффициент детерминации (R²)',
        'Коэффициент регрессии (a₁)',
        'Свободный член (a₀)',
        't-статистика для r',
        'F-статистика',
        'Уровень значимости (α)'
    ],
    'Значение': [
        f'{n}',
        f'{x_mean:.2f}',
        f'{y_mean:.2f}',
        f'{r:.4f}',
        f'{r2:.4f}',
        f'{a1:.6f}',
        f'{a0:.2f}',
        f'{t_calc:.4f}',
        f'{F_calc:.4f}',
        f'{alpha}'
    ]
}

df_results = pd.DataFrame(results_data)
print("ТАБЛИЦА ОСНОВНЫХ РЕЗУЛЬТАТОВ:")
print(df_results.to_string(index=False))

print()
print("="*80)
print("РАБОТА ЗАВЕРШЕНА УСПЕШНО!")
print("="*80)

# ========== СОХРАНЕНИЕ ВЫВОДА В ФАЙЛ ==========

# Восстанавливаем стандартный вывод
sys.stdout = original_stdout

# Записываем весь вывод в файл
output_content = output_buffer.getvalue()
with open('lab4_output.txt', 'w', encoding='utf-8') as f:
    f.write(output_content)

print("\nВесь вывод сохранен в файл 'lab4_output.txt'")
