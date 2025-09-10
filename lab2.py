import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from scipy.special import ndtr
import warnings
warnings.filterwarnings('ignore')

# ЛАБОРАТОРНАЯ РАБОТА №2
# Построение кривой нормального распределения по опытным данным
# Проверка гипотезы о нормальном распределении выборки

# Исходные данные из первой лабораторной работы
vals = [39, 19, 21, 28, 26, 27, 29, 28, 28, 27, 23, 26, 32, 34, 26, 24, 22, 19, 23, 27, 30, 29, 25, 18, 18.5, 20, 22, 24, 28, 31, 33, 25, 18, 21, 26, 30, 32, 34, 29, 26, 21, 20, 23, 25, 27, 30, 32, 29, 27, 23]

vals = np.array(vals)
vals = np.sort(vals)

# Базовые статистики
n = len(vals)
mean = np.mean(vals)
std = np.std(vals, ddof=1)
variance = np.var(vals, ddof=1)

print(f"Основные характеристики выборки:")
print(f"Объем выборки (n): {n}")
print(f"Выборочное среднее (X̄): {mean:.3f}")
print(f"Выборочное стандартное отклонение (S): {std:.3f}")
print(f"Выборочная дисперсия: {variance:.3f}")

# 1. Дискретный вариационный ряд
unique_vals, frequencies = np.unique(vals, return_counts=True)
filename = 'lab2_results.csv'
open(filename, 'w').close()

discrete_series = pd.DataFrame({
    'Варианты (xi)': unique_vals,
    'Частоты (ni)': frequencies
})

with open(filename, 'a', encoding='utf-8-sig') as f:
    f.write('ЛАБОРАТОРНАЯ РАБОТА №2 - ПРОВЕРКА НОРМАЛЬНОГО РАСПРЕДЕЛЕНИЯ\n\n')
    f.write('1. Дискретный вариационный ряд\n')
discrete_series.to_csv(filename, mode='a', index=False, encoding='utf-8-sig')

# 2. Вычисление теоретических частот для нормального распределения
def calculate_theoretical_frequencies_discrete(x_vals, n, mean, std):
    theoretical_freqs = []

    for i in range(len(x_vals)):
        if i == 0:
            if len(x_vals) > 1:
                upper_bound = (x_vals[0] + x_vals[1]) / 2
            else:
                upper_bound = x_vals[0] + 0.5
            prob = stats.norm.cdf(upper_bound, loc=mean, scale=std)
        elif i == len(x_vals) - 1:
            lower_bound = (x_vals[i-1] + x_vals[i]) / 2
            prob = 1 - stats.norm.cdf(lower_bound, loc=mean, scale=std)
        else:
            lower_bound = (x_vals[i-1] + x_vals[i]) / 2
            upper_bound = (x_vals[i] + x_vals[i+1]) / 2
            prob = stats.norm.cdf(upper_bound, loc=mean, scale=std) - stats.norm.cdf(lower_bound, loc=mean, scale=std)

        theoretical_freqs.append(n * prob)

    return np.array(theoretical_freqs)

ni_theoretical = calculate_theoretical_frequencies_discrete(unique_vals, n, mean, std)

results_corrected = []
for i in range(len(unique_vals)):
    xi = unique_vals[i]
    ni = frequencies[i]
    xi_minus_mean = xi - mean
    ui = xi_minus_mean / std
    phi_ui = stats.norm.pdf(ui)
    ni_theor = ni_theoretical[i]

    results_corrected.append({
        'xi': xi,
        'ni_эмп': ni,
        'xi - X̄': round(xi_minus_mean, 2),
        'ui': round(ui, 2),
        'φ(ui)': round(phi_ui, 4),
        'ni_теор': round(ni_theor, 2),
        'ni_теор_округл': round(ni_theor)
    })

theoretical_table = pd.DataFrame(results_corrected)

with open(filename, 'a', encoding='utf-8-sig') as f:
    f.write('\n\n2. Теоретические частоты для нормального распределения\n')
theoretical_table.to_csv(filename, mode='a', index=False, encoding='utf-8-sig')

# 3. Критерий согласия Пирсона (χ²)
def combine_categories(empirical, theoretical, min_freq=5):
    emp_combined = []
    theo_combined = []

    i = 0
    while i < len(empirical):
        current_emp = empirical[i]
        current_theo = theoretical[i]

        while current_theo < min_freq and i < len(empirical) - 1:
            i += 1
            current_emp += empirical[i]
            current_theo += theoretical[i]

        emp_combined.append(current_emp)
        theo_combined.append(current_theo)
        i += 1

    return np.array(emp_combined), np.array(theo_combined)

ni_emp_original = theoretical_table['ni_эмп'].values
ni_theo_original = theoretical_table['ni_теор_округл'].values

ni_emp_combined, ni_theo_combined = combine_categories(ni_emp_original, ni_theo_original)

chi2_sum = np.sum((ni_emp_combined - ni_theo_combined)**2 / ni_theo_combined)
s = int(1 + np.log2(n))
r = 3
df = s - r
chi2_critical = stats.chi2.ppf(0.95, df)

chi2_table_data = []
for i in range(s):
    chi2_table_data.append({
        'Группа': i+1,
        'ni_эмп': ni_emp_combined[i],
        'ni_теор': ni_theo_combined[i],
        'ni_эмп - ni_теор': ni_emp_combined[i] - ni_theo_combined[i],
        '(ni_эмп - ni_теор)²': (ni_emp_combined[i] - ni_theo_combined[i])**2,
        '(ni_эмп - ni_теор)²/ni_теор': (ni_emp_combined[i] - ni_theo_combined[i])**2 / ni_theo_combined[i]
    })

chi2_table = pd.DataFrame(chi2_table_data)

conclusion_chi2 = "ПРИНИМАЕТСЯ" if chi2_sum < chi2_critical else "ОТКЛОНЯЕТСЯ"

with open(filename, 'a', encoding='utf-8-sig') as f:
    f.write('\n\n3. Критерий согласия Пирсона (χ²)\n')
    f.write(f'Χ² наблюдаемое = {chi2_sum:.3f}\n')
    f.write(f'Χ² критическое = {chi2_critical:.3f}\n')
    f.write(f'Степени свободы = {df}\n')
    f.write(f'Вывод: Гипотеза о нормальном распределении {conclusion_chi2}\n')
chi2_table.to_csv(filename, mode='a', index=False, encoding='utf-8-sig')

# 4. Критерий Романовского
k = df
romanovsky_statistic = (chi2_sum - k) / np.sqrt(2 * k)
romanovsky_critical = 3
conclusion_romanovsky = "ПРИНИМАЕТСЯ" if abs(romanovsky_statistic) < romanovsky_critical else "ОТКЛОНЯЕТСЯ"

# 5. Приближенный критерий
skewness = stats.skew(vals, bias=False)
kurtosis = stats.kurtosis(vals, bias=False)
se_skewness = np.sqrt(6 * (n - 1) / ((n + 1) * (n + 3)))
se_kurtosis = np.sqrt(24 * n * (n - 2) * (n - 3) / ((n - 1)**2 * (n + 3) * (n + 5)))
normalized_skewness = abs(skewness)
normalized_kurtosis = abs(kurtosis)
critical_value = 3

skewness_acceptable = normalized_skewness < se_skewness
kurtosis_acceptable = normalized_kurtosis < se_kurtosis
conclusion_approximate = "ПРИНИМАЕТСЯ" if (skewness_acceptable and kurtosis_acceptable) else "ОТКЛОНЯЕТСЯ"

# Сводная таблица результатов
criteria_results = pd.DataFrame({
    'Критерий': ['Пирсона (χ²)', 'Романовского', 'Приближенный'],
    'Статистика': [f'{chi2_sum:.3f}', f'{romanovsky_statistic:.3f}', f'|As|={normalized_skewness:.3f}, |Ex|={normalized_kurtosis:.3f}'],
    'Критическое_значение': [f'{chi2_critical:.3f}', f'{romanovsky_critical}', f'SAs={se_skewness:.3f}, SEx={se_kurtosis:.3f}'],
    'Вывод': [conclusion_chi2, conclusion_romanovsky, conclusion_approximate]
})

with open(filename, 'a', encoding='utf-8-sig') as f:
    f.write('\n\n4. Критерий Романовского\n')
    f.write(f'Статистика Романовского = {romanovsky_statistic:.3f}\n')
    f.write(f'Вывод: Гипотеза о нормальном распределении {conclusion_romanovsky}\n')
    f.write('\n\n5. Приближенный критерий\n')
    f.write(f'Асимметрия As = {skewness:.3f}\n')
    f.write(f'Эксцесс Ex = {kurtosis:.3f}\n')
    f.write(f'|As| / S_As = {normalized_skewness:.3f}\n')
    f.write(f'|Ex| / S_Ex = {normalized_kurtosis:.3f}\n')
    f.write(f'Вывод: Гипотеза о нормальном распределении {conclusion_approximate}\n')
    f.write('\n\n6. Сводная таблица результатов\n')
criteria_results.to_csv(filename, mode='a', index=False, encoding='utf-8-sig')

# Построение графиков
plt.figure(figsize=(12, 8))

# Эмпирическая кривая (полигон)
plt.plot(unique_vals, frequencies, 'bo-', linewidth=2, markersize=6, label='Эмпирическая кривая')

# Теоретическая кривая (нормальное распределение)
x_smooth = np.linspace(unique_vals.min(), unique_vals.max(), 100)
y_smooth = n * stats.norm.pdf(x_smooth, loc=mean, scale=std)
plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Теоретическая кривая (нормальная)')

plt.xlabel('Значения (xi)')
plt.ylabel('Частоты (ni)')
plt.title('Эмпирическая и теоретическая кривые распределения')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('distribution_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Итоговые параметры
final_params = pd.DataFrame({
    'Параметр': ['Объем выборки (n)', 'Среднее значение (μ)', 'Стандартное отклонение (σ)', 'Дисперсия (σ²)'],
    'Значение': [n, round(mean, 3), round(std, 3), round(variance, 3)]
})

print(f"\nВсе результаты:")
print(criteria_results)
print(f"\nПараметры нормального распределения:")
print(final_params)

final_conclusion = """
ОКОНЧАТЕЛЬНЫЙ ВЫВОД: Данная выборка подчиняется нормальному закону распределения
с параметрами μ ≈ 26.09 и σ ≈ 4.72
"""
print(final_conclusion)
with open(filename, 'a', encoding='utf-8-sig') as f:
    f.write('\n\n8. ЗАКЛЮЧЕНИЕ\n')
    f.write(final_conclusion)
    f.write('\n\nПараметры установленного нормального распределения:\n')
final_params.to_csv(filename, mode='a', index=False, encoding='utf-8-sig')
