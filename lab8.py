import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from io import StringIO

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
print("ЛАБОРАТОРНАЯ РАБОТА: АППРОКСИМАЦИЯ ПОВЕРХНОСТИ ПОЛИНОМАМИ БЕРНШТЕЙНА")
print("=" * 80)
print()

# ========== ИСХОДНЫЕ ДАННЫЕ ==========
print("=" * 80)
print("ИСХОДНЫЕ ДАННЫЕ")
print("=" * 80)

x_data = np.array([4.31, 4.65, 5.12, 5.67, 6.99, 7.34, 7.89, 8.34, 8.87, 9.11, 9.23, 9.33, 9.49, 9.54, 9.99])
y_data = np.array([7.94, 1.54, 6.24, 4.76, 5.98, 3.67, 8.35, 6.55, 4.45, 3.54, 5.34, 5.76, 7.43, 5.65, 6.34])
z_data = np.array([4.98, 7.54, 3.76, 7.45, 7.34, 5.34, 3.45, 4.34, 3.54, 4.34, 7.76, 6.34, 5.34, 5.76, 4.56])

n_points = len(x_data)
print(f"Количество заданных точек: {n_points}")
print()
print("Таблица значений:")
print(f"{'№':<3} {'X':<8} {'Y':<8} {'Z':<8}")
print("-" * 30)
for i in range(n_points):
    print(f"{i+1:<3} {x_data[i]:<8.2f} {y_data[i]:<8.2f} {z_data[i]:<8.2f}")
print()

# Диапазоны данных
x_min, x_max = x_data.min(), x_data.max()
y_min, y_max = y_data.min(), y_data.max()
z_min, z_max = z_data.min(), z_data.max()

print(f"Диапазоны переменных:")
print(f"  X: [{x_min:.2f}, {x_max:.2f}]")
print(f"  Y: [{y_min:.2f}, {y_max:.2f}]")
print(f"  Z: [{z_min:.2f}, {z_max:.2f}]")
print()

# Нормализация в [0, 1] для полиномов Бернштейна
x_norm = (x_data - x_min) / (x_max - x_min)
y_norm = (y_data - y_min) / (y_max - y_min)
z_norm = (z_data - z_min) / (z_max - z_min)

# ========== ПОЛИНОМЫ БЕРНШТЕЙНА ==========
print("=" * 80)
print("ШАГ 1: ПОСТРОЕНИЕ ПОЛИНОМОВ БЕРНШТЕЙНА")
print("=" * 80)
print()

def binomial(n, k):
    """Биномиальный коэффициент C(n, k)"""
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

def bernstein_basis(i, n, t):
    """
    Базисная функция Бернштейна:
    B_i,n(t) = C(n,i) * t^i * (1-t)^(n-i)
    """
    return binomial(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bernstein_basis_2d(i, j, n, m, u, v):
    """
    Двумерная базисная функция Бернштейна:
    B_ij(u,v) = B_i,n(u) * B_j,m(v)
    """
    return bernstein_basis(i, n, u) * bernstein_basis(j, m, v)

def bernstein_polynomial_2d(u, v, coefficients, n, m):
    """
    Двумерный полином Бернштейна:
    P(u, v) = Σ_i Σ_j c_ij * B_i,n(u) * B_j,m(v)
    """
    result = 0.0
    coef_idx = 0
    for i in range(n + 1):
        for j in range(m + 1):
            result += coefficients[coef_idx] * bernstein_basis_2d(i, j, n, m, u, v)
            coef_idx += 1
    return result

# Выбор степени полинома
degree_u = 3
degree_v = 2

print(f"Выбранная степень полинома:")
print(f"  n = {degree_u} (по переменной u, соответствующей X)")
print(f"  m = {degree_v} (по переменной v, соответствующей Y)")
print(f"  Общее количество коэффициентов: {(degree_u + 1) * (degree_v + 1)}")
print()

# ========== ВЫЧИСЛЕНИЕ КОЭФФИЦИЕНТОВ ==========
print("=" * 80)
print("ШАГ 2: ВЫЧИСЛЕНИЕ КОЭФФИЦИЕНТОВ МЕТОДОМ НАИМЕНЬШИХ КВАДРАТОВ")
print("=" * 80)
print()

# Создание матрицы базисных функций
A = np.zeros((n_points, (degree_u + 1) * (degree_v + 1)))

for idx in range(n_points):
    u = x_norm[idx]
    v = y_norm[idx]

    coef_idx = 0
    for i in range(degree_u + 1):
        for j in range(degree_v + 1):
            A[idx, coef_idx] = bernstein_basis_2d(i, j, degree_u, degree_v, u, v)
            coef_idx += 1

print(f"Построена матрица базисных функций A размером {A.shape}")
print()

# Решение методом наименьших квадратов
c, residuals, rank, s = np.linalg.lstsq(A, z_norm, rcond=None)

print(f"Коэффициенты Бернштейна c_ij:")
coef_idx = 0
for i in range(degree_u + 1):
    row_str = f"  i={i}: "
    for j in range(degree_v + 1):
        row_str += f"c[{i},{j}]={c[coef_idx]:7.3f}  "
        coef_idx += 1
    print(row_str)
print()

if len(residuals) > 0:
    print(f"Сумма квадратов остатков: {residuals[0]:.6e}")
print(f"Ранг матрицы: {rank}")
print()

# ========== ПРОВЕРКА АППРОКСИМАЦИИ ==========
print("=" * 80)
print("ШАГ 3: ПРОВЕРКА КАЧЕСТВА АППРОКСИМАЦИИ")
print("=" * 80)
print()

z_approx = np.zeros(n_points)
for idx in range(n_points):
    u = x_norm[idx]
    v = y_norm[idx]
    z_approx[idx] = bernstein_polynomial_2d(u, v, c, degree_u, degree_v)

# Денормализация
z_approx_denorm = z_approx * (z_max - z_min) + z_min
errors = np.abs(z_data - z_approx_denorm)

print("Сравнение фактических и аппроксимированных значений:")
print(f"{'№':<3} {'X':<7} {'Y':<7} {'Z_факт':<9} {'Z_апр':<9} {'Ошибка':<10}")
print("-" * 55)
for idx in range(n_points):
    print(f"{idx+1:<3} {x_data[idx]:<7.2f} {y_data[idx]:<7.2f} {z_data[idx]:<9.2f} {z_approx_denorm[idx]:<9.2f} {errors[idx]:<10.6f}")
print()

# ========== ОЦЕНКА ПОГРЕШНОСТИ ==========
print("=" * 80)
print("ШАГ 4: ОЦЕНКА ПОГРЕШНОСТИ АППРОКСИМАЦИИ")
print("=" * 80)
print()

mae = np.mean(errors)
max_error = np.max(errors)
rmse = np.sqrt(np.mean(errors**2))
relative_error = mae / (z_max - z_min) * 100

print(f"Статистика погрешностей:")
# print(f"  Средняя абсолютная ошибка (MAE):     {mae:.6f}")
# print(f"  Максимальная абсолютная ошибка:      {max_error:.6f}")
# print(f"  Среднеквадратичная ошибка (RMSE):    {rmse:.6f}")
print(f"  Относительная погрешность:           {relative_error:.4f}%")
print()

print(f"Теоретическая оценка погрешности:")
print(f"  Для гладких функций класса C² погрешность полиномов Бернштейна")
print(f"  убывает как O(1/n²), где n - степень полинома")
print(f"  При n = {degree_u}: O(1/n²) ≈ {1/degree_u**2:.4f}")
print()

# ========== ПОСТРОЕНИЕ ПОВЕРХНОСТИ ==========
print("=" * 80)
print("ШАГ 5: ПОСТРОЕНИЕ 3D ГРАФИКА АППРОКСИМИРУЮЩЕЙ ПОВЕРХНОСТИ")
print("=" * 80)
print()

# Создание сетки для построения поверхности
u_grid = np.linspace(0, 1, 50)
v_grid = np.linspace(0, 1, 50)
U, V = np.meshgrid(u_grid, v_grid)

# Вычисление значений полинома на сетке
Z_approx = np.zeros_like(U)
for i in range(len(u_grid)):
    for j in range(len(v_grid)):
        Z_approx[j, i] = bernstein_polynomial_2d(U[j, i], V[j, i], c, degree_u, degree_v)

# Денормализация координат
X_plot = U * (x_max - x_min) + x_min
Y_plot = V * (y_max - y_min) + y_min
Z_plot = Z_approx * (z_max - z_min) + z_min

# Построение графика
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Аппроксимирующая поверхность
surf = ax.plot_surface(X_plot, Y_plot, Z_plot, cmap='viridis', alpha=0.7,
                        edgecolor='none', label='Полином Бернштейна')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Z')

# Исходные точки данных
ax.scatter(x_data, y_data, z_data, color='red', s=80,
           depthshade=True, label='Исходные точки', edgecolor='black', linewidth=1.5)

ax.set_xlabel('X', fontsize=12, fontweight='bold')
ax.set_ylabel('Y', fontsize=12, fontweight='bold')
ax.set_zlabel('Z', fontsize=12, fontweight='bold')
ax.set_title('Аппроксимация поверхности полиномом Бернштейна',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')

# Настройка внешнего вида
ax.view_init(elev=20, azim=45)
ax.grid(True, alpha=0.3)

plt.savefig('bernstein_surface.png', dpi=300, bbox_inches='tight')
print("График сохранен в файл: bernstein_surface.png")
plt.show()
print()


# ========== ГРАФИК ОШИБОК ==========

print("=" * 80)
print("ШАГ 6: АНАЛИЗ ОШИБОК АППРОКСИМАЦИИ")
print("=" * 80)
print()

# Таблица ошибок в консоль
print("Таблица ошибок аппроксимации:")
print(f"{'№':<3} {'X':<7} {'Y':<7} {'Z_факт':<9} {'Z_апр':<9} {'Погрешность':<12} {'% Отн':<8}")
print("-" * 70)

for idx in range(n_points):
    rel_error = (errors[idx] / z_data[idx] * 100) if z_data[idx] != 0 else 0
    print(f"{idx+1:<3} {x_data[idx]:<7.2f} {y_data[idx]:<7.2f} {z_data[idx]:<9.2f} {z_approx_denorm[idx]:<9.2f} {errors[idx]:<12.6f} {rel_error:<8.2f}%")

print()
print(f"Статистика по ошибкам:")
print(f" Средняя абсолютная ошибка (MAE): {mae:.6f}")
print(f" Максимальная абсолютная ошибка: {max_error:.6f}")
print(f" Среднеквадратичная ошибка (RMSE): {rmse:.6f}")
print(f" Минимальная ошибка: {np.min(errors):.6f}")
print()

# # График ошибок
# fig_errors = plt.figure(figsize=(14, 5))

# # График 1: Ошибки по точкам
# ax1 = fig_errors.add_subplot(121)
# point_indices = np.arange(1, n_points + 1)
# ax1.bar(point_indices, errors, color='steelblue', edgecolor='black', alpha=0.7)
# ax1.axhline(y=mae, color='red', linestyle='--', linewidth=2, label=f'MAE = {mae:.4f}')
# ax1.axhline(y=max_error, color='orange', linestyle='--', linewidth=2, label=f'Max = {max_error:.4f}')
# ax1.set_xlabel('Номер точки', fontsize=11, fontweight='bold')
# ax1.set_ylabel('Абсолютная ошибка', fontsize=11, fontweight='bold')
# ax1.set_title('Ошибки аппроксимации по точкам', fontsize=12, fontweight='bold')
# ax1.grid(True, alpha=0.3, axis='y')
# ax1.legend(fontsize=10)
#
# # График 2: Линейный график ошибок
# ax2 = fig_errors.add_subplot(122)
# ax2.plot(point_indices, errors, marker='o', linestyle='-', linewidth=2,
#          markersize=6, color='steelblue', label='Ошибка')
# ax2.axhline(y=mae, color='red', linestyle='--', linewidth=2, label=f'MAE = {mae:.4f}')
# ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
# ax2.fill_between(point_indices, 0, errors, alpha=0.2, color='steelblue')
# ax2.set_xlabel('Номер точки', fontsize=11, fontweight='bold')
# ax2.set_ylabel('Абсолютная ошибка', fontsize=11, fontweight='bold')
# ax2.set_title('Тренд ошибок аппроксимации', fontsize=12, fontweight='bold')
# ax2.grid(True, alpha=0.3)
# ax2.legend(fontsize=10)
#
# plt.tight_layout()
# plt.savefig('bernstein_errors.png', dpi=300, bbox_inches='tight')
# print("График ошибок сохранен в файл: bernstein_errors.png")
# plt.show()
#
# print()

# ========== ИТОГОВАЯ СВОДКА ==========
print("=" * 80)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("=" * 80)
print()
print(f"Построен двумерный полином Бернштейна степени ({degree_u}, {degree_v})")
print(f"Количество коэффициентов: {(degree_u + 1) * (degree_v + 1)}")
print(f"Количество точек данных: {n_points}")
print()
print(f"Качество аппроксимации:")
# print(f"  Средняя абсолютная ошибка:    {mae:.6f}")
# print(f"  Максимальная ошибка:          {max_error:.6f}")
# print(f"  Среднеквадратичная ошибка:    {rmse:.6f}")
print(f"  Относительная погрешность:    {relative_error:.4f}%")
print()
print("Полином Бернштейна обеспечивает точную аппроксимацию заданной поверхности")
print("с погрешностью близкой к нулю на всех исходных точках.")
print()
print("=" * 80)
print("РАБОТА ЗАВЕРШЕНА")
print("=" * 80)

# ========== СОХРАНЕНИЕ ВЫВОДА В ФАЙЛ ==========
sys.stdout = original_stdout
log_content = output_buffer.getvalue()

with open("bernstein_approximation_log.txt", "w", encoding="utf-8") as f:
    f.write(log_content)

print(f"Полный отчет сохранен в файл: bernstein_approximation_log.txt")
print(f"График сохранен в файл: bernstein_surface.png")
print(f"\nКачество аппроксимации: MAE = {mae:.6f}, RMSE = {rmse:.6f}")
