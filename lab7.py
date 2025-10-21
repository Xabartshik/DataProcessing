import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from io import StringIO
# Чтобы высчитать многомерный полином лагранжа, нужно сделать следующее:
# 1. Выбрать опорные точки рядом с выбранной координатой (создать опорную сетку)
# 2. Для каждого из выбранных Х строится полином 1-мерный для каждого Y
# 3. После этого, расчитать
# ========== НАСТРОЙКА ВЫВОДА В ФАЙЛ ==========
original_stdout = sys.stdout
output_buffer = StringIO()

class Logger:
    def __init__(self, *files):
        self.files = files
    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Дублируем вывод в консоль и буфер
sys.stdout = Logger(original_stdout, output_buffer)

print("=" * 80)
print("ЛАБОРАТОРНАЯ РАБОТА: ДВУМЕРНАЯ ИНТЕРПОЛЯЦИЯ МЕТОДОМ ЛАГРАНЖА")
print("Вариант: 11-15")
print("=" * 80)
print()

# ========== ИСХОДНЫЕ ДАННЫЕ ==========
print("=" * 80)
print("ИСХОДНЫЕ ДАННЫЕ")
print("=" * 80)

x_nodes = np.array([0.35, 0.55, 0.75, 0.95, 1.15, 1.35, 1.55, 1.75, 1.95])
y_nodes = np.array([7.15, 7.45, 7.75, 8.05, 8.35, 8.65, 8.95])
z_data = np.array([
    [7.87, 4.17, 8.65, 3.19, 4.65, 5.98, 7.12, 6.65, 3.76],
    [4.75, 3.76, 2.19, 5.34, 4.65, 6.14, 7.54, 3.33, 6.54],
    [5.43, 4.24, 5.43, 4.83, 5.33, 3.54, 5.34, 4.32, 3.43],
    [6.33, 3.33, 5.43, 4.54, 5.84, 5.34, 4.54, 5.43, 4.54],
    [4.43, 8.35, 5.43, 4.43, 9.54, 3.43, 5.32, 8.34, 6.54],
    [5.54, 8.65, 4.54, 3.54, 4.76, 3.76, 7.65, 4.54, 5.76],
    [3.54, 8.95, 5.76, 7.54, 5.96, 3.23, 5.76, 4.34, 3.54]
])

print("Узлы по оси X:")
print(f"  {x_nodes}")
print(f"Узлы по оси Y:")
print(f"  {y_nodes}")
print(f"\nТаблица значений функции z(x,y):")
print(f"{'Y\\X':<8}", end="")
for x_val in x_nodes:
    print(f"{x_val:>8.2f}", end="")
print()
for i, y_val in enumerate(y_nodes):
    print(f"{y_val:<8.2f}", end="")
    for j in range(len(x_nodes)):
        print(f"{z_data[i,j]:>8.2f}", end="")
    print()
print()

# Точка интерполяции
x0, y0 = 0.98, 8.28
print(f"Требуется найти значение функции в точке: ({x0}, {y0})")
print()

# Границы производных
M_x = 4  # все производные по x ограничены числом 4
M_y = 2  # все производные по y ограничены числом 2
print(f"Ограничения на производные:")
print(f"  |∂ⁿz/∂xⁿ| ≤ {M_x} для всех n")
print(f"  |∂ⁿz/∂yⁿ| ≤ {M_y} для всех n")
print()

# ========== МЕТОД ЛАГРАНЖА ==========
print("=" * 80)
print("ШАГ 1: ВЫБОР ЛОКАЛЬНОЙ ОБЛАСТИ ИНТЕРПОЛЯЦИИ")
print("=" * 80)

def lagrange_1d(x, y, t):
    """Одномерная интерполяция полиномом Лагранжа."""
    n = len(x)
    L = 0.0
    for i in range(n):
        li = 1.0
        for j in range(n):
            if j != i:
                li *= (t - x[j]) / (x[i] - x[j])
        L += y[i] * li
    return L

# Выбор 4 ближайших узлов для построения полинома 3-й степени
n_nodes = 4
x_indices = np.argsort(np.abs(x_nodes - x0))[:n_nodes]
y_indices = np.argsort(np.abs(y_nodes - y0))[:n_nodes]
x_indices = np.sort(x_indices)
y_indices = np.sort(y_indices)

x_sub = x_nodes[x_indices]
y_sub = y_nodes[y_indices]
z_sub = z_data[y_indices, :][:, x_indices]

print(f"Выбраны {n_nodes} ближайших узла по X: {x_sub}")
print(f"Выбраны {n_nodes} ближайших узла по Y: {y_sub}")
print(f"\nЗначения функции в опорных узлах:")
print(f"{'Y\\X':<8}", end="")
for x_val in x_sub:
    print(f"{x_val:>8.2f}", end="")
print()
for i, y_val in enumerate(y_sub):
    print(f"{y_val:<8.2f}", end="")
    for j in range(n_nodes):
        print(f"{z_sub[i,j]:>8.2f}", end="")
    print()
print()

# ========== ИНТЕРПОЛЯЦИЯ ==========
print("=" * 80)
print("ШАГ 2: ПОСЛОЙНАЯ ИНТЕРПОЛЯЦИЯ МЕТОДОМ ЛАГРАНЖА")
print("=" * 80)

print(f"\nА) Интерполяция по оси X при фиксированных Y:")
interp_values = []
for k, y_i in enumerate(y_sub):
    row_vals = z_sub[k, :]
    val = lagrange_1d(x_sub, row_vals, x0)
    interp_values.append(val)
    print(f"   y = {y_i:.2f}: z({x0}, {y_i:.2f}) = {val:.6f}")

print(f"\nБ) Интерполяция по оси Y с использованием промежуточных значений:")
z_lagrange = lagrange_1d(y_sub, interp_values, y0)
print(f"   Итоговое значение: z({x0}, {y0}) = {z_lagrange:.6f}")
print()

# ========== ОЦЕНКА ПОГРЕШНОСТИ ==========
print("=" * 80)
print("ШАГ 3: ОЦЕНКА ПОГРЕШНОСТИ ИНТЕРПОЛЯЦИИ")
print("=" * 80)

h_x = x_sub[-1] - x_sub[0]
h_y = y_sub[-1] - y_sub[0]

print(f"\nРазмах опорной сетки:")
print(f"  h_x = {x_sub[-1]:.2f} - {x_sub[0]:.2f} = {h_x:.2f}")
print(f"  h_y = {y_sub[-1]:.2f} - {y_sub[0]:.2f} = {h_y:.2f}")
print()

# Оценка погрешности для полинома Лагранжа степени (n-1)
# Для полинома степени 3 в каждом направлении, остаточный член содержит 4-ю производную
# Простая верхняя оценка: |R| ≤ (h^4 / (4!)) * M для одномерного случая
# Для двумерного используем приближенную оценку как сумму вкладов от x и y

# Максимальное расстояние от точки интерполяции до ближайших узлов
dx_max = max(abs(x0 - x_sub[0]), abs(x0 - x_sub[-1]))
dy_max = max(abs(y0 - y_sub[0]), abs(y0 - y_sub[-1]))

print(f"Максимальные расстояния от точки интерполяции до опорных узлов:")
print(f"  Δx_max = {dx_max:.4f}")
print(f"  Δy_max = {dy_max:.4f}")
print()


error_x = (h_x**n_nodes / math.factorial(n_nodes)) * M_x
error_y = (h_y**n_nodes / math.factorial(n_nodes)) * M_y
error_bound = error_x + error_y

print(f"Оценка погрешности по остаточному члену полинома Лагранжа:")
print(f"  Вклад от X: (h_x^{n_nodes} / {n_nodes}!) × M_x = ({h_x:.2f}^{n_nodes} / {math.factorial(n_nodes)}) × {M_x} = {error_x:.6f}")
print(f"  Вклад от Y: (h_y^{n_nodes} / {n_nodes}!) × M_y = ({h_y:.2f}^{n_nodes} / {math.factorial(n_nodes)}) × {M_y} = {error_y:.6f}")
print(f"\n  ИТОГОВАЯ ОЦЕНКА ПОГРЕШНОСТИ: |δz| ≤ {error_bound:.6f}")
print()

# ========== ПОСТРОЕНИЕ 3D ПОВЕРХНОСТИ ==========
print("=" * 80)
print("ШАГ 4: ПОСТРОЕНИЕ 3D ГРАФИКА ИНТЕРПОЛЯЦИОННОЙ ПОВЕРХНОСТИ")
print("=" * 80)

# Создаем плотную сетку для построения
x_surf = np.linspace(x_sub.min(), x_sub.max(), 40)
y_surf = np.linspace(y_sub.min(), y_sub.max(), 40)
X_surf, Y_surf = np.meshgrid(x_surf, y_surf)
Z_surf = np.zeros_like(X_surf)

# Вычисляем Z в каждой точке плотной сетки
for i in range(len(y_surf)):
    for j in range(len(x_surf)):
        interp_vals = [lagrange_1d(x_sub, z_sub[k, :], X_surf[i, j]) for k in range(n_nodes)]
        Z_surf[i, j] = lagrange_1d(y_sub, interp_vals, Y_surf[i, j])

# Построение графика
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Интерполяционная поверхность
surf = ax.plot_surface(X_surf, Y_surf, Z_surf, cmap='viridis', alpha=0.8, edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='z')

# Опорные узлы
X_sub_grid, Y_sub_grid = np.meshgrid(x_sub, y_sub)
ax.scatter(X_sub_grid, Y_sub_grid, z_sub, color='red', s=60, depthshade=True, label='Опорные узлы (4×4)')

# Точка интерполяции
ax.scatter(x0, y0, z_lagrange, color='cyan', s=250, marker='*', edgecolor='black', linewidth=2, label=f'Точка ({x0}, {y0})')

ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z', fontsize=12)
ax.set_title('Интерполяционная поверхность методом Лагранжа', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)

plt.savefig('lagrange_interpolation_3d.png', dpi=300, bbox_inches='tight')
print("График сохранен в файл: lagrange_interpolation_3d.png")
plt.show()
print()

# ========== ИТОГОВАЯ СВОДКА ==========
print("=" * 80)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("=" * 80)
print(f"\nТочка интерполяции: ({x0}, {y0})")
print(f"Значение функции: z({x0}, {y0}) = {z_lagrange:.6f}")
print(f"\nОценка погрешности:")
print(f"  По остаточному члену Лагранжа: |δz| ≤ {error_bound*100:.4f}%")
print(f"\nИспользованные опорные узлы: {n_nodes}×{n_nodes} = {n_nodes**2} точек")
print(f"Степень полинома Лагранжа: {n_nodes-1} в каждом направлении")
print()
print("=" * 80)
print("РАБОТА ЗАВЕРШЕНА")
print("=" * 80)

# ========== СОХРАНЕНИЕ ПОЛНОГО ВЫВОДА В ФАЙЛ ==========
sys.stdout = original_stdout
log_content = output_buffer.getvalue()

with open("lagrange_interpolation_log.txt", "w", encoding="utf-8") as f:
    f.write(log_content)
print(f"Значение функции в точке ({x0}, {y0}): {z_lagrange:.6f}")
print(f"Оценка погрешности: |δz| ≤ {error_bound:.6f}")
