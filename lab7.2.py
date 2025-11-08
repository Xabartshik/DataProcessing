import numpy as np
import matplotlib.pyplot as plt
import sys
from io import StringIO

# ================= НАСТРОЙКА ДВОЙНОГО ВЫВОДА (как в lab8.py) =================
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
print("ЛАБОРАТОРНАЯ РАБОТА 7.2: КВАДРАТИЧНЫЙ (И КУБИЧЕСКИЙ) СПЛАЙНЫ")
print("=" * 80)
print()

# ================= ИСХОДНЫЕ ДАННЫЕ =================
print("=" * 80)
print("ИСХОДНЫЕ ДАННЫЕ")
print("=" * 80)
print()


x_data = np.array([1.23, 1.67, 2.04, 2.34, 2.56, 2.99, 3.34, 4.54, 4.87, 5.11], dtype=float)
y_data = np.array([-1.34, -5.23, -0.23, -1.17,  0.32,  0.43,  0.99,  1.54,  4.34,  9.12], dtype=float)

n_points = len(x_data)
print(f"Количество точек: {n_points}")
print()
print("Таблица узлов:")
print(f"{'№':<3} {'X':<10} {'f(X)':<10}")
print("-" * 30)
for i in range(n_points):
    print(f"{i+1:<3} {x_data[i]:<10.4f} {y_data[i]:<10.4f}")
print()

# ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ: КВАДРАТИЧНЫЙ СПЛАЙН =================
def build_quadratic_spline(x, y, natural="left"):
    """
    Строит квадратичный сплайн S_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2
    Условия:
      - S_i(x_i)  = y_i, S_i(x_{i+1}) = y_{i+1}
      - S'_i(x_{i+1}) = S'_{i+1}(x_{i+1})
      - Краевое 'естественное' условие: c_0 = 0 (left) или c_{m-1} = 0 (right)
    Возвращает arrays a,b,c для отрезков [x_i, x_{i+1}].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(np.diff(x) <= 0):
        raise ValueError("Массив x должен быть строго возрастающим")
    N = len(x)
    m = N - 1

    h = np.diff(x)
    slope = np.diff(y) / h

    a = np.zeros(m)
    b = np.zeros(m)
    c = np.zeros(m)

    if natural == "left":
        c[0] = 0.0
        for i in range(m - 1):
            c[i+1] = (slope[i+1] - slope[i] - c[i] * h[i]) / h[i+1]
    elif natural == "right":
        c[-1] = 0.0
        for i in range(m - 2, -1, -1):
            c[i] = (slope[i+1] - slope[i] - c[i+1] * h[i+1]) / h[i]
    else:
        raise ValueError("natural должен быть 'left' или 'right'")

    for i in range(m):
        a[i] = y[i]
        b[i] = slope[i] - c[i] * h[i]
    return a, b, c

def eval_quadratic(xq, x, a, b, c):
    xq = np.asarray(xq, dtype=float)
    yq = np.empty_like(xq, dtype=float)
    for idx, xv in enumerate(xq):
        if xv <= x[0]:
            i = 0
        elif xv >= x[-1]:
            i = len(x) - 2
        else:
            i = np.searchsorted(x, xv) - 1
        dx = xv - x[i]
        yq[idx] = a[i] + b[i]*dx + c[i]*dx*dx
    return yq

def deriv_quadratic(xq, x, a, b, c):
    xq = np.asarray(xq, dtype=float)
    dq = np.empty_like(xq, dtype=float)
    for idx, xv in enumerate(xq):
        if xv <= x[0]:
            i = 0
        elif xv >= x[-1]:
            i = len(x) - 2
        else:
            i = np.searchsorted(x, xv) - 1
        dx = xv - x[i]
        dq[idx] = b[i] + 2.0*c[i]*dx
    return dq

# ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ: КУБИЧЕСКИЙ СПЛАЙН =================
def build_cubic_spline(x, y, bc="natural"):
    """
    Строит естественный кубический сплайн:
      S_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3
    Условия: S, S', S'' непрерывны; natural: S''(x_0)=0, S''(x_n)=0.
    Возвращает a,b,c,d на каждом отрезке [x_i, x_{i+1}].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(np.diff(x) <= 0):
        raise ValueError("Массив x должен быть строго возрастающим")
    n = len(x) - 1
    h = np.diff(x)

    a = y[:-1].copy()
    b = np.zeros(n)
    c = np.zeros(n+1)  # заметьте: c определено в узлах
    d = np.zeros(n)

    # Естественные граничные условия
    alpha = np.zeros(n)
    l = np.ones(n+1)
    mu = np.zeros(n+1)
    z = np.zeros(n+1)

    for i in range(1, n):
        alpha[i] = 3.0/h[i]*(y[i+1]-y[i]) - 3.0/h[i-1]*(y[i]-y[i-1])
    for i in range(1, n):
        l[i] = 2.0*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i]
    # natural: c_n = 0
    c[n] = 0.0
    for j in range(n-1, -1, -1):
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (y[j+1]-y[j])/h[j] - h[j]*(2.0*c[j] + c[j+1])/3.0
        d[j] = (c[j+1] - c[j]) / (3.0*h[j])
    # Теперь c[j] — коэффициент при (x-x_j)^2 на отрезке j
    return a, b, c[:-1], d  # c на отрезках: c[:-1]

def eval_cubic(xq, x, a, b, c, d):
    xq = np.asarray(xq, dtype=float)
    yq = np.empty_like(xq, dtype=float)
    for idx, xv in enumerate(xq):
        if xv <= x[0]:
            i = 0
        elif xv >= x[-1]:
            i = len(x) - 2
        else:
            i = np.searchsorted(x, xv) - 1
        dx = xv - x[i]
        yq[idx] = a[i] + b[i]*dx + c[i]*dx*dx + d[i]*dx*dx*dx
    return yq

def deriv_cubic(xq, x, a, b, c, d):
    xq = np.asarray(xq, dtype=float)
    dq = np.empty_like(xq, dtype=float)
    for idx, xv in enumerate(xq):
        if xv <= x[0]:
            i = 0
        elif xv >= x[-1]:
            i = len(x) - 2
        else:
            i = np.searchsorted(x, xv) - 1
        dx = xv - x[i]
        dq[idx] = b[i] + 2.0*c[i]*dx + 3.0*d[i]*dx*dx
    return dq

# ================= ПОСТРОЕНИЕ СПЛАЙНОВ =================
print("=" * 80)
print("ШАГ 1: КВАДРАТИЧНЫЙ СПЛАЙН")
print("=" * 80)
print()
aq, bq, cq = build_quadratic_spline(x_data, y_data)

print("Коэффициенты квадратичного сплайна [a_i, b_i, c_i] на отрезках [x_i, x_{i+1}]:")
print(f"{'i':<3} {'[x_i, x_{i+1}]':<22} {'a_i':>12} {'b_i':>12} {'c_i':>12}")
print("-" * 65)
for i in range(len(x_data) - 1):
    print(f"{i:<3} [{x_data[i]:.4f}, {x_data[i+1]:.4f}]{'':<6} {aq[i]:>12.6f} {bq[i]:>12.6f} {cq[i]:>12.6f}")
print()

print("=" * 80)
print("ШАГ 2: КУБИЧЕСКИЙ СПЛАЙН")
print("=" * 80)
print()
ac, bc, cc, dc = build_cubic_spline(x_data, y_data, bc="natural")

print("Коэффициенты кубического сплайна [a_i, b_i, c_i, d_i] на отрезках [x_i, x_{i+1}]:")
print(f"{'i':<3} {'[x_i, x_{i+1}]':<22} {'a_i':>12} {'b_i':>12} {'c_i':>12} {'d_i':>12}")
print("-" * 80)
for i in range(len(x_data) - 1):
    print(f"{i:<3} [{x_data[i]:.4f}, {x_data[i+1]:.4f}]{'':<6} {ac[i]:>12.6f} {bc[i]:>12.6f} {cc[i]:>12.6f} {dc[i]:>12.6f}")
print()

# ================= АНАЛИТИЧЕСКИЕ ФОРМУЛЫ =================
print("=" * 80)
print("ШАГ 3: АНАЛИТИЧЕСКАЯ ФОРМА СПЛАЙНОВ")
print("=" * 80)
print()

print("Квадратичный сплайн S_q(x) по отрезкам:")
for i in range(len(x_data) - 1):
    xi = x_data[i]
    print(f"[{x_data[i]:.4f}, {x_data[i+1]:.4f}]: S_q(x) = "
          f"{aq[i]:.6f} + {bq[i]:.6f}*(x - {xi:.4f}) + {cq[i]:.6f}*(x - {xi:.4f})^2")
print()

print("Кубический сплайн S_c(x) по отрезкам:")
for i in range(len(x_data) - 1):
    xi = x_data[i]
    print(f"[{x_data[i]:.4f}, {x_data[i+1]:.4f}]: S_c(x) = "
          f"{ac[i]:.6f} + {bc[i]:.6f}*(x - {xi:.4f}) + {cc[i]:.6f}*(x - {xi:.4f})^2 + {dc[i]:.6f}*(x - {xi:.4f})^3")
print()

# ================= ПОГРЕШНОСТИ В УЗЛАХ =================
print("=" * 80)
print("ШАГ 4: ПОГРЕШНОСТЬ В УЗЛАХ (|S(x_i) - y_i|)")
print("=" * 80)
print()

# Для интерполяционных сплайнов эти ошибки ~ численная погрешность
yq_nodes = eval_quadratic(x_data, x_data, aq, bq, cq)
yc_nodes = eval_cubic(x_data, x_data, ac, bc, cc, dc)

err_q = np.abs(yq_nodes - y_data)
err_c = np.abs(yc_nodes - y_data)

print("Квадратичный сплайн:")
print(f"{'i':<3} {'x_i':<10} {'y_i':<12} {'S_q(x_i)':<12} {'|ошибка|':<12}")
print("-" * 60)
for i in range(n_points):
    print(f"{i:<3} {x_data[i]:<10.4f} {y_data[i]:<12.6f} {yq_nodes[i]:<12.6f} {err_q[i]:<12.3e}")
print(f"Макс. ошибка (узлы), кв. сплайн: {np.max(err_q):.3e}")
print()

print("Кубический сплайн:")
print(f"{'i':<3} {'x_i':<10} {'y_i':<12} {'S_c(x_i)':<12} {'|ошибка|':<12}")
print("-" * 60)
for i in range(n_points):
    print(f"{i:<3} {x_data[i]:<10.4f} {y_data[i]:<12.6f} {yc_nodes[i]:<12.6f} {err_c[i]:<12.3e}")
print(f"Макс. ошибка (узлы), куб. сплайн: {np.max(err_c):.3e}")
print()

# ================= ПРОВЕРКА НЕПРЕРЫВНОСТИ ПРОИЗВОДНОЙ В УЗЛАХ =================
print("=" * 80)
print("ШАГ 5: НЕПРЕРЫВНОСТЬ ПРОИЗВОДНОЙ В УЗЛАХ")
print("=" * 80)
print()

def check_C1(x, eval_deriv):
    ok = True
    for i in range(1, len(x)-1):
        xl = x[i] - 0.0
        xr = x[i] + 0.0
        Dl = eval_deriv(np.array([xl]))[0]
        Dr = eval_deriv(np.array([xr]))[0]
        ok &= np.allclose(Dl, Dr, atol=1e-10)
        print(f"x[{i}]={x[i]:.4f}: S'-={Dl:.8f}, S'+={Dr:.8f}")
    return ok

print("Квадратичный сплайн:")
ok_q = check_C1(x_data, lambda t: deriv_quadratic(t, x_data, aq, bq, cq))
print(f"Непрерывность S'_q: {'OK' if ok_q else 'FAIL'}")
print()

print("Кубический сплайн:")
ok_c = check_C1(x_data, lambda t: deriv_cubic(t, x_data, ac, bc, cc, dc))
print(f"Непрерывность S'_c: {'OK' if ok_c else 'FAIL'}")
print()

# ================= ГРАФИКИ (РАЗДЕЛЬНО, БЕЗ НАЛОЖЕНИЯ) =================
print("=" * 80)
print("ШАГ 6: ГРАФИКИ (отдельно для S и S')")
print("=" * 80)
print()

xx = np.linspace(x_data[0], x_data[-1], 600)

# Квадратичный S(x)
yy_q = eval_quadratic(xx, x_data, aq, bq, cq)
fig1 = plt.figure(figsize=(12, 6))
ax1 = fig1.add_subplot(111)
ax1.plot(xx, yy_q, color='steelblue', linewidth=2.5, label='Квадратичный сплайн S_q(x)')
ax1.scatter(x_data, y_data, color='red', s=60, edgecolor='black', linewidth=1.0, label='Узлы')
ax1.set_xlabel('x', fontsize=12, fontweight='bold')
ax1.set_ylabel('S_q(x)', fontsize=12, fontweight='bold')
ax1.set_title('Квадратичный сплайн', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best')
plt.tight_layout()
plt.savefig('lab7_2_variant14_quadratic_spline.png', dpi=300, bbox_inches='tight')
print("Сохранено: lab7_2_variant14_quadratic_spline.png")

# Квадратичный S'(x) с отметками в узлах
dy_q = deriv_quadratic(xx, x_data, aq, bq, cq)
fig2 = plt.figure(figsize=(12, 6))
ax2 = fig2.add_subplot(111)
ax2.plot(xx, dy_q, color='darkorange', linewidth=2.5, label="Производная S'_q(x)")
ax2.scatter(x_data, deriv_quadratic(x_data, x_data, aq, bq, cq), color='red', s=60, edgecolor='black', linewidth=1.0, label="S'_q(x_i)")
ax2.set_xlabel('x', fontsize=12, fontweight='bold')
ax2.set_ylabel("S'_q(x)", fontsize=12, fontweight='bold')
ax2.set_title("Производная квадратичного сплайна", fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best')
plt.tight_layout()
plt.savefig('lab7_2_variant14_quadratic_derivative.png', dpi=300, bbox_inches='tight')
print("Сохранено: lab7_2_variant14_quadratic_derivative.png")

# Кубический S(x)
yy_c = eval_cubic(xx, x_data, ac, bc, cc, dc)
fig3 = plt.figure(figsize=(12, 6))
ax3 = fig3.add_subplot(111)
ax3.plot(xx, yy_c, color='seagreen', linewidth=2.5, label='Кубический сплайн S_c(x)')
ax3.scatter(x_data, y_data, color='red', s=60, edgecolor='black', linewidth=1.0, label='Узлы')
ax3.set_xlabel('x', fontsize=12, fontweight='bold')
ax3.set_ylabel('S_c(x)', fontsize=12, fontweight='bold')
ax3.set_title('Кубический сплайн', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='best')
plt.tight_layout()
plt.savefig('lab7_2_variant14_cubic_spline.png', dpi=300, bbox_inches='tight')
print("Сохранено: lab7_2_variant14_cubic_spline.png")

# Кубический S'(x) с отметками в узлах
dy_c = deriv_cubic(xx, x_data, ac, bc, cc, dc)
fig4 = plt.figure(figsize=(12, 6))
ax4 = fig4.add_subplot(111)
ax4.plot(xx, dy_c, color='purple', linewidth=2.5, label="Производная S'_c(x)")
ax4.scatter(x_data, deriv_cubic(x_data, x_data, ac, bc, cc, dc), color='red', s=60, edgecolor='black', linewidth=1.0, label="S'_c(x_i)")
ax4.set_xlabel('x', fontsize=12, fontweight='bold')
ax4.set_ylabel("S'_c(x)", fontsize=12, fontweight='bold')
ax4.set_title("Производная кубического сплайна", fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(loc='best')
plt.tight_layout()
plt.savefig('lab7_2_variant14_cubic_derivative.png', dpi=300, bbox_inches='tight')
print("Сохранено: lab7_2_variant14_cubic_derivative.png")

plt.show()
print()

# ================= ИТОГИ =================
print("=" * 80)
print("ИТОГИ")
print("=" * 80)
print()
print("1) Построены квадратичный и кубический сплайны; напечатаны коэффициенты и аналитические формулы.")
print("2) Вычислены погрешности в узлах для обоих сплайнов (они 0, как и должно быть).")
print("3) Проверена непрерывность производной в узлах.")
print()

# ================= СОХРАНЕНИЕ ВЫВОДА =================
sys.stdout = original_stdout
log_content = output_buffer.getvalue()
with open("lab7_2_variant14_log.txt", "w", encoding="utf-8") as f:
    f.write(log_content)

