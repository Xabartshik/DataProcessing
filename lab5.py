# ЛАБОРАТОРНАЯ РАБОТА № 5. ВАРИАНТ 14
# Построение модельного уравнения нелинейной регрессии
# Выбор между y = a + b/x и y = 1/(a x + b), оценка параметров, индекс корреляции, F-критерий
# Оформление и логирование вывода выполнены в стиле lab4.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
from io import StringIO
import os

# ========== НАСТРОЙКА ВЫВОДА В ФАЙЛ (как в lab4) ==========
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

print("=== ЛАБОРАТОРНАЯ РАБОТА № 5. ВАРИАНТ 14 ===")
print("Тема: Построение модельного уравнения нелинейной регрессии; выбор гиперболической модели; проверка адекватности")
print()

# ========== ИСХОДНЫЕ ДАННЫЕ ==========
# Вариант №14: размер предприятия X (млн. руб. основных средств) и себестоимость Y (руб.) единицы продукции
x = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7.5], dtype=float)
y = np.array([15.0, 11.0, 12.0, 10.8, 10.0, 9.0, 8.0], dtype=float)
n = len(x)
assert n == len(y)

print("Исходные наблюдения (X, Y):")
for xi, yi in zip(x, y):
    print(f"  X = {xi:.3f}, Y = {yi:.3f}")
print(f"n = {n}")
print()

# ========== ШАГ 1: КОРРЕЛЯЦИОННОЕ ПОЛЕ ==========
print("=== ШАГ 1: ПОСТРОЕНИЕ КОРРЕЛЯЦИОННОГО ПОЛЯ ===")
print("Подготовлены точки для визуализации рассеяния (X против Y)")
print()

# ========== ШАГ 2: ПОДГОТОВКА ДЛЯ ДВУХ МОДЕЛЕЙ И МЕТОД КОНЕЧНЫХ РАЗНОСТЕЙ ==========
print("=== ШАГ 2: ВЫБОР ВИДА ФУНКЦИИ ПО МЕТОДУ КОНЕЧНЫХ РАЗНОСТЕЙ ===")
# Модель A: y = a + b/x  => линейнизация: Y1 = x*y, X1 = x, Y1 = a*X1 + b
X1 = x.copy()
Y1 = x * y

# Модель B: y = 1/(a x + b) => линейнизация: Y2 = 1/y, X2 = x, Y2 = a*X2 + b
X2 = x.copy()
Y2 = 1.0 / y

def diff_ratios(X, Y):
    # Отношения ΔY/ΔX для последовательных пар (в порядке возрастания X)
    ratios = []
    for i in range(len(X)-1):
        dX = X[i+1] - X[i]
        dY = Y[i+1] - Y[i]
        if dX != 0:
            ratios.append(dY / dX)
    return np.array(ratios, dtype=float)

rat1 = diff_ratios(X1, Y1)
rat2 = diff_ratios(X2, Y2)

def variability_score(r):
    # Используем стандартное отклонение отношений как меру разброса
    return float(np.std(r, ddof=1))

score1 = variability_score(rat1)
score2 = variability_score(rat2)

print("Отношения ΔY/ΔX для модели A (Y=xy):", ", ".join(f"{v:.5f}" for v in rat1))
print("Отношения ΔY/ΔX для модели B (Y=1/y):", ", ".join(f"{v:.5f}" for v in rat2))
print(f"Мера вариабельности отношений (std): A = {score1:.6f}, B = {score2:.6f}")

# Выбор лучшей модели по меньшей вариабельности конечных разностей
chosen = "B" if score2 < score1 else "A"
print(f"Выбрана модель: {'y = 1/(a x + b)' if chosen=='B' else 'y = a + b/x'}")
print()

# ========== ШАГ 3: ОЦЕНКА ПАРАМЕТРОВ МНК ДЛЯ ОБЕИХ МОДЕЛЕЙ ==========
print("=== ШАГ 3: ОЦЕНКА ПАРАМЕТРОВ МНК ДЛЯ ОБЕИХ МОДЕЛЕЙ ===")

def ols_linear(X, Y):
    Sx = np.sum(X)
    Sy = np.sum(Y)
    Sxx = np.sum(X*X)
    Sxy = np.sum(X*Y)
    det = n * Sxx - Sx**2
    a = (n * Sxy - Sx * Sy) / det       # наклон
    b = (Sy - a * Sx) / n               # свободный член
    return a, b, Sx, Sy, Sxx, Sxy, det

# Модель A (Y1 = a*X1 + b) => y_hat = a + b/x
aA, bA, Sx1, Sy1, Sxx1, Sxy1, det1 = ols_linear(X1, Y1)

# Модель B (Y2 = a*X2 + b) => y_hat = 1/(a*x + b)
aB, bB, Sx2, Sy2, Sxx2, Sxy2, det2 = ols_linear(X2, Y2)

print("Суммы (модель A, Y=xy):")
print(f"  ΣX={Sx1:.6f}, ΣY={Sy1:.6f}, ΣX²={Sxx1:.6f}, ΣXY={Sxy1:.6f}, det={det1:.6f}")
print(f"Параметры (модель A в исходном виде y=a+b/x): a={aA:.6f}, b={bA:.6f}")

print("Суммы (модель B, Y=1/y):")
print(f"  ΣX={Sx2:.6f}, ΣY={Sy2:.6f}, ΣX²={Sxx2:.6f}, ΣXY={Sxy2:.6f}, det={det2:.6f}")
print(f"Параметры (модель B в исходном виде y=1/(a x + b)): a={aB:.6f}, b={bB:.6f}")
print()

# ========== ШАГ 4: ПРОГНОЗЫ, ОСТАТКИ И КРИТЕРИИ ДЛЯ ОБЕИХ МОДЕЛЕЙ ==========
print("=== ШАГ 4: ПРОГНОЗЫ, ОСТАТКИ, КОРРЕЛЯЦИОННЫЙ ИНДЕКС И F-КРИТЕРИЙ ===")
y_bar = float(np.mean(y))

def predict_A(xv):  # y = a + b/x
    return aA + bA / xv

def predict_B(xv):  # y = 1/(a x + b)
    return 1.0 / (aB * xv + bB)

y_hat_A = predict_A(x)
y_hat_B = predict_B(x)

def diagnostics(y_true, y_hat):
    TSS = float(np.sum((y_true - np.mean(y_true))**2))
    RSS = float(np.sum((y_true - y_hat)**2))
    ESS = float(np.sum((y_hat - np.mean(y_true))**2))
    eta = np.sqrt(ESS / TSS) if TSS > 0 else 0.0
    k1 = 1
    k2 = len(y_true) - 2
    F_calc = (ESS / k1) / (RSS / k2) if RSS > 0 else np.inf
    return TSS, ESS, RSS, eta, k1, k2, F_calc

TSS_A, ESS_A, RSS_A, eta_A, k1_A, k2_A, F_A = diagnostics(y, y_hat_A)
TSS_B, ESS_B, RSS_B, eta_B, k1_B, k2_B, F_B = diagnostics(y, y_hat_B)

alpha = 0.05
Fcrit_A = stats.f.ppf(1 - alpha, k1_A, k2_A)
Fcrit_B = stats.f.ppf(1 - alpha, k1_B, k2_B)

print("Модель A (y = a + b/x):")
print(f"  TSS={TSS_A:.6f}, ESS={ESS_A:.6f}, RSS={RSS_A:.6f}")
print(f"  Индекс корреляции η = sqrt(ESS/TSS) = {eta_A:.6f}")
print(f"  F_набл = {F_A:.6f}, F_крит(α={alpha}, {k1_A},{k2_A}) = {Fcrit_A:.6f}")
print(f"  Вывод по Фишеру: {'адекватна' if F_A > Fcrit_A else 'неадекватна'}")

print("Модель B (y = 1/(a x + b)):")
print(f"  TSS={TSS_B:.6f}, ESS={ESS_B:.6f}, RSS={RSS_B:.6f}")
print(f"  Индекс корреляции η = sqrt(ESS/TSS) = {eta_B:.6f}")
print(f"  F_набл = {F_B:.6f}, F_крит(α={alpha}, {k1_B},{k2_B}) = {Fcrit_B:.6f}")
print(f"  Вывод по Фишеру: {'адекватна' if F_B > Fcrit_B else 'неадекватна'}")
print()

# Итоговый выбор (если отличается от шага 2, ориентируемся на методичку — конечные разности как основной шаг выбора)
chosen = "B" if chosen == "B" else "A"  # оставляем выбор по конечным разностям
print(f"Итоговая модель для представления результатов: {'y = 1/(a x + b)' if chosen=='B' else 'y = a + b/x'}")
if chosen == "B":
    a_ch, b_ch = aB, bB
    y_hat = y_hat_B
    eta_ch, F_ch, Fcrit_ch, k2_ch = eta_B, F_B, Fcrit_B, k2_B
else:
    a_ch, b_ch = aA, bA
    y_hat = y_hat_A
    eta_ch, F_ch, Fcrit_ch, k2_ch = eta_A, F_A, Fcrit_A, k2_A

print(f"Параметры итоговой модели: a = {a_ch:.6f}, b = {b_ch:.6f}")
print(f"Индекс корреляции η = {eta_ch:.6f}")
print(f"F_набл = {F_ch:.6f} при степенях свободы (1, {k2_ch}), F_крит = {Fcrit_ch:.6f}")
print()

# ========== ШАГ 5: ГРАФИК ==========
print("=== ШАГ 5: ГРАФИК КОРРЕЛЯЦИОННОГО ПОЛЯ И ЛИНИИ РЕГРЕССИИ ===")
os.makedirs("figures", exist_ok=True)
plt.figure(figsize=(12, 6))
plt.scatter(x, y, c="#444", s=50, alpha=0.8, label="Наблюдения")

xx = np.linspace(np.min(x)*0.9, np.max(x)*1.1, 400)
yyA = predict_A(xx)
yyB = predict_B(xx)

# Основная линия — выбранная модель; альтернативная — пунктиром
if chosen == "B":
    plt.plot(xx, yyB, "r-", lw=2.5, label="y = 1/(a x + b)")
    plt.plot(xx, yyA, "g--", lw=1.8, alpha=0.7, label="y = a + b/x (сравнение)")
else:
    plt.plot(xx, yyA, "r-", lw=2.5, label="y = a + b/x")
    plt.plot(xx, yyB, "g--", lw=1.8, alpha=0.7, label="y = 1/(a x + b) (сравнение)")

plt.xlabel("X (млн. руб. основных средств)")
plt.ylabel("Y (руб.) себестоимость единицы")
plt.title("ЛР5 Вариант 14: Корреляционное поле и линия регрессии")
plt.grid(True, alpha=0.3)
plt.legend()

fig_path = os.path.join("figures", "lab5_variant14.png")
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"График сохранен: {fig_path}")
print()

# ========== ИТОГОВАЯ СВОДКА ==========
print("=== ИТОГОВАЯ СВОДКА ===")
print(f"Выбранная модель: {'y = 1/(a x + b)' if chosen=='B' else 'y = a + b/x'}")
print(f"Параметры: a = {a_ch:.6f}, b = {b_ch:.6f}")
print(f"Индекс корреляции (корреляционное отношение): η = {eta_ch:.6f}")
print(f"Критерий Фишера: F_набл = {F_ch:.6f} {'>' if F_ch > Fcrit_ch else '<='} F_крит = {Fcrit_ch:.6f} -> модель {'адекватна' if F_ch > Fcrit_ch else 'неадекватна'}")
print()

# ========== СОХРАНЕНИЕ ПОЛНОГО ВЫВОДА В ФАЙЛ ==========
sys.stdout = original_stdout
with open("lab5_output.txt", "w", encoding="utf-8") as f:
    f.write(output_buffer.getvalue())
print("Готово: график сохранен в figures/, полный консольный вывод записан в lab5_output.txt")
