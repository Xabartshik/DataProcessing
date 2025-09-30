import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import pandas as pd


def get_y_on_x_coeffs(y_bar, x_bar, r, std_y, std_x):
    """
    Вычисляет коэффициенты уравнения регрессии y на x по формуле:
    y_hat = y_bar + r * (std_y / std_x) * (x - x_bar).
    Возвращает кортеж (a0, a1), где y = a0 + a1 * x.
    """
    # Коэффициент при x (a1)
    a1 = r * (std_y / std_x)

    # Свободный член (a0) через раскрытие скобок
    a0 = y_bar - a1 * x_bar

    return a0, a1


def get_x_on_y_coeffs(x_bar, y_bar, r, std_x, std_y):
    """
    Вычисляет коэффициенты уравнения регрессии x на y по формуле:
    x_hat = x_bar + r * (std_x / std_y) * (y - y_bar).
    Возвращает кортеж (b0, b1), где x = b0 + b1 * y.
    """
    # Коэффициент при y (b1)
    b1 = r * (std_x / std_y)

    # Свободный член (b0) через раскрытие скобок
    b0 = x_bar - b1 * y_bar

    return b0, b1

X = np.array([9, 13, 17, 22, 29, 36, 44, 51, 60, 65])
Y = np.array([27, 36, 29, 41, 54, 71, 65, 81, 90, 95])

n = len(X)


with open('lab3_variant14.txt', 'w', encoding='utf-8') as f:
    f.write("Лабораторная работа №3. Вариант 14\n\n")
    f.write("Исходные данные (Таблица 23):\n")
    data_df = pd.DataFrame({'X': X, 'Y': Y})
    f.write(data_df.to_string(index=False) + "\n\n")

    # 1. Построить корреляционное поле и выбрать вид регрессии
    f.write("1. Построение корреляционного поля.\n")
    f.write("По расположению точек связь выглядит линейной.\n\n")
    plt.scatter(X, Y, label='Данные')
    slope, intercept = np.polyfit(X, Y, 1)
    plt.plot(X, slope * X + intercept, color='red', label='Линия тренда')
    plt.xlabel('X (реализация продукции, млн. руб.)')
    plt.ylabel('Y (накладные расходы, тыс. руб.)')
    plt.title('Корреляционное поле')
    plt.legend()
    plt.savefig('correlation_field.png')
    plt.close()

    # 2. Вычислить числовые характеристики x_bar, y_bar, Sx, Sy, r, sigma_r
    f.write("2. Вычисление числовых характеристик.\n")
    x_bar = np.mean(X)
    y_bar = np.mean(Y)
    f.write(f"Реализация продукции (x_bar) = {x_bar:.2f}\n")
    f.write(f"Накладные расходы (y_bar) = {y_bar:.2f}\n")

    # Таблица 24
    df = pd.DataFrame({
        'xi': X,
        'xi - x_bar': X - x_bar,
        '(xi - x_bar)^2': (X - x_bar)**2,
        'yi': Y,
        'yi - y_bar': Y - y_bar,
        '(yi - y_bar)^2': (Y - y_bar)**2,
        'x^2': X**2,
        'xy': X * Y
    })
    sums = df.sum(numeric_only=True)
    df.loc['Сумма'] = sums
    f.write("Расчетная таблица 24:\n")
    f.write(df.to_string() + "\n\n")

    Sx2 = sums['(xi - x_bar)^2'] / (n - 1)
    Sx = np.sqrt(Sx2)
    Sy2 = sums['(yi - y_bar)^2'] / (n - 1)
    Sy = np.sqrt(Sy2)
    f.write(f"Sx^2 = {Sx2:.3f}, Sx = {Sx:.2f}\n")
    f.write(f"Sy^2 = {Sy2:.3f}, Sy = {Sy:.2f}\n")

    xy_bar = np.mean(X * Y)
    r = (xy_bar - x_bar * y_bar) / (Sx * Sy)
    f.write(f"r = {r:.2f}\n\n")

    # 3. Определить значимость коэффициента корреляции r и доверительный интервал
    f.write("3. Проверка значимости коэффициента корреляции r.\n")
    tp = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
    f.write(f"tp = {tp:.2f}\n")
    alpha = 0.05
    df_freedom = n - 2
    t_crit = t.ppf(1 - alpha/2, df_freedom)
    f.write(f"t_табл (alpha=0.05, df={df_freedom}) = {t_crit:.2f}\n")
    if tp > t_crit:
        f.write("Коэффициент корреляции значим, поскольку tp > t_табл.\n")
    else:
        f.write("Коэффициент корреляции не значим.\n")

    f.write("Доверительный интервал для r с gamma=0.95.\n")
    sigma_r = (1 - r**2) / np.sqrt(n - 2)
    f.write(f"sigma_r = {sigma_r:.2f}\n")
    t_gamma = 1.96  # Из таблицы Лапласа для gamma=0.95
    lower = r - t_gamma * sigma_r
    upper = min(r + t_gamma * sigma_r, 1)  # Кап на 1
    f.write(f"r ∈ [{lower:.2f}; {upper:.2f}]\n\n")

    # 4. Эмпирические уравнения линий регрессий
    f.write("4. Эмпирические уравнения регрессий.\n")
    a0, a1 = get_y_on_x_coeffs(y_bar, x_bar, r, Sy, Sx)
    f.write(f"y на x: y = {a0:.2f} + {a1:.2f} x\n")

    b0, b1 = get_x_on_y_coeffs(x_bar, y_bar, r, Sx, Sy)
    f.write(f"x на y: x = {b0:.2f} + {b1:.2f} y\n")

    # Контроль
    control = a1 * b1
    f.write(f"Контроль: r^2 = {r**2:.2f}, a1 * b1 = {control:.2f}\n\n")

    # 5. Коэффициент детерминации R2
    f.write("5. Коэффициент детерминации R2.\n")
    R2 = r**2
    f.write(f"R2 = {R2:.2f}\n")
    f.write("Смысловое значение: {R2*100:.0f}% вариации Y объясняется вариацией X.\n\n")

    # 6. Проверка адекватности уравнения регрессии y на x
    f.write("6. Проверка адекватности уравнения регрессии y на x.\n")
    f.write("Таблица 25. Остатки и предсказанные значения.\n")
    f.write("xi    yi    y_hat_x    y_i - y_hat_x\n")
    for i in range(n):
        y_hat_x = a0 + a1 * X[i]
        residual = Y[i] - y_hat_x
        f.write(f"{X[i]:5.1f}  {Y[i]:5.1f}  {y_hat_x:8.6f}  {residual:12.6f}\n")
    f.write("\n")
    # Предсказанные значения y_hat для всех точек
    y_hat = a0 + a1 * X

    # Сумма квадратов остатков
    ss_res = np.sum((Y - y_hat) ** 2)

    # Общая сумма квадратов (вокруг среднего)
    ss_tot = np.sum((Y - y_bar) ** 2)

    # Коэффициент детерминации R^2
    R2 = 1 - (ss_res / ss_tot)
    f.write(f"Сумма квадратов остатков (SS_res) = {ss_res:.2f}\n")
    f.write(f"Общая сумма квадратов (SS_tot) = {ss_tot:.2f}\n")
    f.write(f"Коэффициент детерминации R^2 = {R2:.2f}\n")
    f.write(f"Смысловое значение: {R2 * 100:.0f}% вариации Y объясняется вариацией X.\n\n")

    # Проверка адекватности через F-тест
    df_num = 1  # Степени свободы числителя (количество регрессионных коэффициентов - 1)
    df_den = n - 2  # Степени свободы знаменателя (n - 2)
    F = (R2 / (1 - R2)) * (df_den / df_num)
    from scipy.stats import f as f_dist

    alpha = 0.05
    F_crit = f_dist.ppf(1 - alpha, df_num, df_den)
    f.write(f"F-статистика = {F:.2f}, F_табл (alpha=0.05, df1={df_num}, df2={df_den}) = {F_crit:.2f}\n")
    if F > F_crit:
        f.write("Уравнение регрессии адекватно, поскольку F > F_табл.\n")
    else:
        f.write("Уравнение регрессии не адекватно.\n")

    # 7. Оценка погрешности уравнения регрессии y на x и его коэффициентов
    f.write("7. Оценка погрешности уравнения регрессии y на x и коэффициентов.\n")
    Y_hat = a0 + a1 * X
    residuals = Y - Y_hat
    u_bar = np.mean(residuals)
    # Создаем таблицу 26 с колонками: u_i, u_i - u_bar, (u_i - u_bar)^2
    u_diff = residuals - u_bar
    u_diff_sq = u_diff ** 2
    f.write("Таблица 26. Расчет отклонений и их квадратов.\n")
    f.write("u_i        u_i - ȳ_u    (u_i - ȳ_u)^2\n")
    for ui, ud, udsq in zip(residuals, u_diff, u_diff_sq):
        f.write(f"{ui:8.2f}  {ud:8.2f}  {udsq:12.4f}\n")
    sum_u_diff_sq = np.sum(u_diff_sq)
    f.write(f"{'':8}  {'':8}  {sum_u_diff_sq:12.4f}\n\n")

    # Средняя квадратическая ошибка по формуле из таблицы 26
    sigma_u = np.sqrt(sum_u_diff_sq / (n - 2))
    relative_error = (sigma_u / np.mean(Y)) * 100
    f.write(f"Средняя квадратическая ошибка σᵤ = {sigma_u:.2f}\n")
    f.write(f"Относительная ошибка δ = {relative_error:.2f}%\n")

    # Погрешности коэффициентов регрессии (по формулам из документа)
    sum_xx = np.sum((X - x_bar) ** 2)

    # Находим S_y/x
    S_yx = np.sqrt(np.sum((Y - Y_hat) ** 2) / (n - 2))
    # По формуле S_a0 и S_a1 через S_y/x и параметры выборки
    S_a0 = S_yx * np.sqrt(np.sum(X ** 2) / (n * sum_xx))
    S_a1 = S_yx / np.sqrt(sum_xx)

    f.write(f"Среднеквадратическая ошибка коэффициента a0 (S_a0) = {S_a0:.2f}\n")
    f.write(f"Среднеквадратическая ошибка коэффициента a1 (S_a1) = {S_a1:.2f}\n")

    # Отношения ошибок к самим коэффициентам
    ratio_a0 = S_a0 / abs(a0)
    ratio_a1 = S_a1 / abs(a1)
    f.write(f"Отношение S_a0/|a0| = {ratio_a0:.2f}\n")
    f.write(f"Отношение S_a1/|a1| = {ratio_a1:.2f}\n")

    # Значимость коэффициентов на основании отношения < 0.5
    if ratio_a0 < 0.5:
        f.write("Коэффициент a0 значим.\n")
    if ratio_a1 < 0.5:
        f.write("Коэффициент a1 значим.\n\n")

    # 8. Уравнение регрессии в первоначальной системе координат
    f.write("8. Уравнение регрессии y на x в первоначальной системе координат.\n")
    f.write(f"y = {a0:.2f} + {a1:.2f} x\n")  # Уже в оригинальных координатах, поскольку данные не масштабированы
    # Значимость коэффициентов регрессии
    if ratio_a0 < 0.5:
        f.write("Коэффициент a0 имеет относительно низкую погрешность и считается значимым.\n")
    else:
        f.write("Коэффициент a0 имеет высокую погрешность, что может указывать на его незначимость.\n")

    if ratio_a1 < 0.5:
        f.write("Коэффициент a1 имеет относительно низкую погрешность и является статистически значимым.\n")
    else:
        f.write("Коэффициент a1 имеет высокую погрешность, что уменьшает его значимость в модели.\n")

    # Адекватность модели (например, F-тест или коэффициент детерминации)
    f.write(f"Коэффициент детерминации R^2 = {R2:.3f}, что указывает на долю вариации Y, объясненную моделью.\n")
    if R2 > 0.7:
        f.write(
            "Модель регрессии объясняет большую часть вариации зависимой переменной, что говорит о хорошей адекватности.\n")
    elif R2 > 0.4:
        f.write("Модель объясняет значимую часть вариации, но существует заметная дисперсия, не объяснённая моделью.\n")
    else:
        f.write("Модель объясняет малую часть вариации, что может свидетельствовать о низкой адекватности.\n")

    # Если в коде есть проверка значимости самой модели через F-тест, можно добавить:
    if F > F_crit:
        f.write(f"F-тест: F = {F:.2f} > критическое значение F_crit = {F_crit:.2f}. Модель значимо описывает данные.\n")
    else:
        f.write(f"F-тест: F = {F:.2f} <= критическое значение F_crit = {F_crit:.2f}. Модель может быть неадекватной.\n")

    # Итог о применимости модели
    if ratio_a1 < 0.5 and R2 > 0.4 and F > F_crit:
        f.write(
            "Итог: Уравнение регрессии y на x является статистически значимым и пригодно для описания зависимости.\n\n")
    else:
        f.write("Итог: Не все показатели указывают на значимость модели. Требуется пересмотреть данные или модель.\n\n")


