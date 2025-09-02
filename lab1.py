import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

filename = 'file.csv'
open(filename, 'w')
vals = [39, 19, 21, 28, 26, 27, 29, 28, 28, 27, 23, 26, 32, 34, 26, 24, 22, 19, 23, 27, 30, 29, 25, 18, 18.5, 20, 22, 24, 28, 31, 33, 25, 18, 21, 26, 30, 32, 34, 29, 26, 21, 20, 23, 25, 27, 30, 32, 29, 27, 23]

vals = np.array(vals)
vals = np.sort(vals)

# Уникальные дискретные значения и их частота
unique, frequency = np.unique(vals, return_counts=True)
print(f'Контр.сумма: {sum(frequency)}, ожидали 50')
discrete_vals = pd.DataFrame({'Значение': unique, 'Частота': frequency})
discrete_vals.to_csv(filename, mode='a')
print(discrete_vals)

plt.figure(figsize = (20, 20))

plt.subplot(2, 2, 1)
plt.grid(True)
plt.bar(unique, frequency)
plt.title('Дискретный ряд')
plt.ylabel('Частота')
plt.xlabel('Значение')
# Интервалы
n = vals.size
k = int(1+np.log2(n))
hist, bin_edges = np.histogram(vals, bins = k)
interval_vals = pd.DataFrame({'Интервал': [f'{bin_edges[i]:.1f}, {bin_edges[i+1]}' for i in range(len(bin_edges)-1)],
                                'Частота': hist})
interval_vals.to_csv(filename, mode='a')
print(interval_vals)
plt.subplot(2, 2, 2)
plt.grid(True)
plt.hist(vals, bins = k, edgecolor = 'black')
plt.title('Гиста и полигон')
plt.ylabel('Частота')
plt.xlabel('Значение')
plt.xticks(bin_edges, bin_edges)

bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
plt.subplot(2, 2, 2)
plt.grid(True)
plt.plot(bin_centers, hist, marker = 'o')

# Кумулята

cum = np.cumsum(frequency)
plt.subplot(2, 2, 3)
plt.title('CUM')
plt.ylabel('Частота')
plt.xticks(unique)
plt.grid(True)
plt.plot(unique, cum/cum[-1], marker = 'o')


# Эмперическая функция
empf = np.cumsum(frequency)/n
plt.subplot(2, 2, 4)
plt.grid(True)
plt.plot(unique, cum/cum[-1], linestyle = '--', marker = None)
plt.step(unique, empf, '<--', where = 'post')
plt.title('Эмперическая функция')
plt.xlabel('Значения')
plt.ylabel('Значения F')

# Мода
mode = stats.mode(vals, keepdims=True).mode[0]
print(f"\nМода: {mode}")

# Медиана
median = np.median(vals)
print(f"Медиана: {median}")

# Выборочное среднее
mean = np.mean(vals)
print(f"Выборочное среднее: {mean:.2f}")

# Выборочное дисперсия
variance = np.var(vals, ddof=1)
print(f"Выборочная дисперсия: {variance:.2f}")

# СКО
std = np.std(vals, ddof=1)
print(f"Выборочное среднее сквадратическое отклонение: {std:.2f}")

# Коэффицент вариации
cv = (std / mean) * 100
print(f"Коэффициент вариации: {cv:.2f}%")

# Ассиметрия
skew = stats.skew(vals)
print(f"Ассиметрия: {skew:.2f}")

# Экcцess
kurt = stats.kurtosis(vals)
print(f"Эксцесс: {kurt:.2f}")

# Доверительные интервалы
confidence_level = 0.95
alpha = 1 - confidence_level

# Для среднего
t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
mean_ci_lower = mean - t_crit * (std / np.sqrt(n))
mean_ci_upper = mean + t_crit * (std / np.sqrt(n))
print(f"\nДИ для среднего: ({mean_ci_lower:.2f}, {mean_ci_upper:.2f})")

# Для дисперсии
chi2_lower = stats.chi2.ppf(alpha/2, df=n-1)
chi2_upper = stats.chi2.ppf(1 - alpha/2, df=n-1)
var_ci_lower = (n-1) * variance / chi2_upper
var_ci_upper = (n-1) * variance / chi2_lower
std_ci_lower = np.sqrt(var_ci_lower)
std_ci_upper = np.sqrt(var_ci_upper)
print(f"ДИ для дисперсии: ({std_ci_lower:.2f}, {std_ci_upper:.2f})")


# Создание словаря
statistics_dict = {
    "Мода": mode,
    "Медиана": median,
    "Выборочное среднее": round(mean, 2),
    "Выборочная дисперсия": round(variance, 2),
    "Выборочное среднее сквадратическое отклонение": round(std, 2),
    "Коэффициент вариации": f"{round(cv, 2)}%",
    "Ассиметрия": round(skew, 2),
    "Эксцесс": round(kurt, 2),
    "ДИ для среднего": (round(mean_ci_lower, 2), round(mean_ci_upper, 2)),
    "ДИ для дисперсии": (round(std_ci_lower, 2), round(std_ci_upper, 2))
}

relative_frequencies = frequency / n
cumulative_frequencies = np.cumsum(frequency)
cumulative_relative_frequencies = cumulative_frequencies / n

mean = np.mean(vals)
std = np.std(vals, ddof=1)
conditional_variants = (unique - mean) / std

conditional_variants_df = pd.DataFrame({
    'Значение (x_i)': unique,
    'Условные варианты (u_i)': conditional_variants,
    'Частоты (n_i)': frequency,
    'Относительные частоты (w_i)': relative_frequencies,
    'Кумулятивные частоты (N_i)': cumulative_frequencies,
    'Кумулятивные относительные частоты (W_i)': cumulative_relative_frequencies
})
conditional_variants_df.to_csv(filename, mode='a')


ecdf_df = pd.DataFrame({
    'F(x)': cumulative_relative_frequencies,
    'x': unique
})

# Add a row for x < min(data) where F(x) = 0
ecdf_df = pd.concat([pd.DataFrame({'F(x)': [0.0], 'x': [min(unique) - 0.1] }), ecdf_df], ignore_index=True)
ecdf_df = ecdf_df.sort_values(by='x')
ecdf_df.to_csv(filename, mode='a')


stat_vals = pd.DataFrame({'Название': statistics_dict.keys(), 'Значение': statistics_dict.values()})
stat_vals.to_csv(filename, mode='a')

# 6. Пояснения
print("\nПояснения:")
print("- Мода: 26 -- чаще всего наблюдаемый дебет газа скважины.")
print("- Медиана: 26 -- скважина вырабатывает либо больше 26 в половине случаев, либо меньше в половине случаев.")
print("- Выборочное среднее: В среднем скважина вырабатывает 26 в день, для оценки общей произвордительност.")
print("- Выборочная дисперсия: Мера разброса данных вокруг среднего -- неслабый разброс значений в дебетах скважины (неоднородность).")
print("- Выборочное стандартное отклонение: Квадратный корень из дисперсии, в тех же единицах, что и данные, в среднем разброс +- 4 единицы в выдаче газа.")
print("- Коэффициент вариации: То же самое, что и отклонение, но в процентах.")
print("- Асимметрия: Показывает, что в среднем скважина выдает больше среднего значения в день (т.к. полож, то правостороняя).")
print("- Эксцесс: Много экстримальных значений (сильно отличающихся от среднего), показывает настбильность")
print("- Доверительный интервал для среднего: Показывает, что средний дебит будет лежать в 24.75-27.43 с вероятностью 95.")
print("- Доверительный интервал для стандартного отклонения: Показывает, что СКО будет лежать в (3.94, 5.88) с вероятностью 95..")

# Left-hand side of the equation
sum_ni = np.sum(frequency)  # Σn_i
sum_ni_ui = np.sum(frequency * conditional_variants)  # Σn_i u_i
sum_ni_ui2 = np.sum(frequency * conditional_variants**2)  # Σn_i u_i^2
left_side = sum_ni + 2 * sum_ni_ui + sum_ni_ui2

# Right-hand side of the equation
sum_ni_ui_plus_1_squared = np.sum(frequency * (conditional_variants + 1)**2)  # Σn_i (u_i + 1)^2
right_side = sum_ni_ui_plus_1_squared

# Print control results
print(f"Контроль вычислений:")
print(f"Σn_i: {sum_ni}")
print(f"Σn_i u_i: {sum_ni_ui:.4f}")
print(f"Σn_i u_i^2: {sum_ni_ui2:.4f}")
print(f"Левая часть (Σn_i + 2Σn_i u_i + Σn_i u_i^2): {left_side:.4f}")
print(f"Правая часть (Σn_i (u_i + 1)^2): {right_side:.4f}")

control_results = pd.DataFrame({
    'Σn_i': sum_ni,
    'Σn_i u_i': sum_ni_ui,
    'Σn_i u_i^2': sum_ni_ui2,
    'Левая часть (Σn_i + 2Σn_i u_i + Σn_i u_i^2)': left_side,
    'Правая часть (Σn_i (u_i + 1)^2)': right_side,
    'Правильность суммы': "Верна" if left_side == right_side else "Неверена"
})
control_results.to_csv(filename, mode='a')

# Check if the control is satisfied
if abs(left_side - right_side) < 1e-10:  # Small tolerance for floating-point comparison
    print("Контроль пройден: обе части уравнения равны.")
else:
    print("Контроль не пройден: разница между частями уравнения значительна.")









plt.savefig(fname = "file.png")
# plt.show()