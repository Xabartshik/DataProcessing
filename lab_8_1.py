import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2


# ============================================================================
# ЛАБОРАТОРНАЯ РАБОТА № 8, ВАРИАНТ 14
# Имитация шестимерного векторного случайного процесса
# с нормально распределенными компонентами:
# M_i(t) = i + t, D_i(t) = t / i, i = 1, ..., 6
# ============================================================================

def math_expect(s):
    """Выборочное математическое ожидание"""
    return sum(s) / len(s)


def dispersion(s, M=None):
    """Выборочная дисперсия (при известном или неизвестном M)"""
    if M is None:
        M = math_expect(s)
    n = len(s)
    D = 0.0
    for x in s:
        D += (x - M) ** 2
    return D / n


def white_noise():
    """Имитация белого шума ~ N(0, 1) по схеме '12 равномерных - 6'"""
    s = 0.0
    for _ in range(12):
        s += random.random()
    return s - 6.0


def normal_val(M, D):
    """Значение нормальной СВ с заданным M и D через белый шум"""
    return white_noise() * math.sqrt(D) + M


# ============================================================================
# ЧАСТЬ 1: ИМИТАЦИЯ ПРОЦЕССА (ЛР8)
# ============================================================================

print("=" * 70)
print("ЛАБОРАТОРНАЯ РАБОТА № 8, ВАРИАНТ 14")
print("Имитация шестимерного векторного случайного процесса")
print("=" * 70)

# номера компонент
I = [i for i in range(1, 7)]
# моменты времени
T = [t for t in range(1, 1001)]

# vals[i-1][t_index] = значение ξ_i(t)
vals = [[] for _ in I]

print("\nГенерация выборки...")
# генерация выборки
for t in T:
    for i in I:
        M_it = i + t  # M_i(t) = i + t
        D_it = t / i  # D_i(t) = t / i
        # vals[i - 1].append(normal_val(M_it, D_it))
        val = np.random.normal(loc=M_it, scale=np.sqrt(D_it))
        vals[i - 1].append(val)

print("Выборка сгенерирована.")

# оценка M_i и D_i по времени для каждой компоненты
comp_means = [math_expect(vals[i - 1]) for i in I]
comp_dispersions = [
    dispersion(vals[i - 1], comp_means[i - 1]) for i in I
]

# M(t), D(t) по всем компонентам
Mt = []
Dt = []
for t_index in range(len(T)):
    snapshot = [vals[i - 1][t_index] for i in I]
    m = math_expect(snapshot)
    d = dispersion(snapshot, m)
    Mt.append(m)
    Dt.append(d)

print(f"\nОценки компонент:")
for i in I:
    print(f"  ξ_{i}: M ≈ {comp_means[i - 1]:.2f}, D ≈ {comp_dispersions[i - 1]:.2f}")

# построение графиков в матрице 2x2
fig, axs = plt.subplots(2, 2, constrained_layout=True)

# M_i по компонентам (верхний левый)
axs[0, 0].plot(I, comp_means, 'o-', color='blue', linewidth=2, markersize=8, label='Оценка M_i')
axs[0, 0].set_title('Математическое ожидание по компонентам', fontsize=12, fontweight='bold')
axs[0, 0].set_xlabel('компонента, i', fontsize=11)
axs[0, 0].set_ylabel('M_i', fontsize=11)
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].legend(loc='best')

# M(t) по времени (верхний правый)
axs[0, 1].plot(T, Mt, color='green', linewidth=1, alpha=0.7, label='M(t)')
axs[0, 1].set_title('Математическое ожидание по времени', fontsize=12, fontweight='bold')
axs[0, 1].set_xlabel('время, t', fontsize=11)
axs[0, 1].set_ylabel('M(t)', fontsize=11)
axs[0, 1].grid(True, alpha=0.3)
axs[0, 1].legend(loc='best')

# D_i по компонентам (нижний левый)
axs[1, 0].plot(I, comp_dispersions, 's-', color='red', linewidth=2, markersize=8, label='Оценка D_i')
axs[1, 0].set_title('Дисперсия по компонентам', fontsize=12, fontweight='bold')
axs[1, 0].set_xlabel('компонента, i', fontsize=11)
axs[1, 0].set_ylabel('D_i', fontsize=11)
axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].legend(loc='best')

# D(t) по времени (нижний правый)
axs[1, 1].plot(T, Dt, color='purple', linewidth=1, alpha=0.7, label='D(t)')
axs[1, 1].set_title('Дисперсия по времени', fontsize=12, fontweight='bold')
axs[1, 1].set_xlabel('время, t', fontsize=11)
axs[1, 1].set_ylabel('D(t)', fontsize=11)
axs[1, 1].grid(True, alpha=0.3)
axs[1, 1].legend(loc='best')

plt.savefig('lab8_variant14_graphs.png', dpi=150, bbox_inches='tight')
print("\nГрафики ЛР8 сохранены в 'lab8_variant14_graphs.png'")
plt.show()

# ============================================================================
# ЧАСТЬ 2: КРИТЕРИЙ ПИРСОНА (ЛР2) ДЛЯ ПРОВЕРКИ НОРМАЛЬНОСТИ
# ============================================================================

print("\n" + "=" * 70)
print("ПРОВЕРКА НОРМАЛЬНОСТИ ПО КРИТЕРИЮ ПИРСОНА (ЛР2)")
print("=" * 70)

alpha = 0.05  # уровень значимости

for component_idx in range(len(I)):
    component_num = I[component_idx]
    sample = np.array(vals[component_idx])
    n = len(sample)

    print(f"\n--- Компонента ξ_{component_num} ---")
    print(f"Объем выборки: n = {n}")

    # 1. Оценки параметров нормального распределения
    m_hat = sample.mean()
    s_hat = sample.std(ddof=1)

    print(f"Оценки параметров:")
    print(f"  m̂ = {m_hat:.4f}")
    print(f"  σ̂ = {s_hat:.4f}")

    # 2. Интервальное разбиение (10 интервалов)
    k_bins = 10
    edges = np.linspace(sample.min(), sample.max(), k_bins + 1)
    obs, _ = np.histogram(sample, bins=edges)

    # 3. Теоретические вероятности и ожидаемые частоты
    p = []
    for j in range(k_bins):
        a, b = edges[j], edges[j + 1]
        p_j = norm.cdf((b - m_hat) / s_hat) - norm.cdf((a - m_hat) / s_hat)
        p.append(p_j)

    p = np.array(p)
    exp = n * p

    # 4. Проверка условия: объединяем интервалы с exp < 5
    mask = exp >= 5
    obs_adj = obs[mask]
    exp_adj = exp[mask]
    k_adj = mask.sum()

    print(f"\nРазбиение: {k_bins} интервалов, после объединения: {k_adj}")
    print(f"Интервалы (выборочная и ожидаемая частоты):")
    print(f"  {'Интервал':<25} {'n_j':<8} {'n\'_j':<10}")
    print("  " + "-" * 45)
    for j in range(k_bins):
        if mask[j]:
            print(f"  [{edges[j]:7.2f}; {edges[j + 1]:7.2f}) {obs[j]:8d} {exp[j]:10.2f}")

    # 5. Статистика Пирсона
    chi2_stat = np.sum((obs_adj - exp_adj) ** 2 / exp_adj)

    # 6. Число степеней свободы
    df = k_adj - 3

    # 7. Критическое значение
    chi2_crit = chi2.ppf(1 - alpha, df)

    print(f"\nКритерий Пирсона:")
    print(f"  χ²_набл = {chi2_stat:.4f}")
    print(f"  χ²_крит (α={alpha}, df={df}) = {chi2_crit:.4f}")

    if chi2_stat <= chi2_crit:
        print(f"  ✓ Результат: χ²_набл ≤ χ²_крит")
        print(f"  ✓ Гипотеза H₀ о нормальности НЕ ОТВЕРГАЕТСЯ")
    else:
        print(f"  ✗ Результат: χ²_набл > χ²_крит")
        print(f"  ✗ Гипотеза H₀ о нормальности ОТВЕРГАЕТСЯ")

print("\n" + "=" * 70)
print("ПРОВЕРКА ЗАВЕРШЕНА")
print("=" * 70)
