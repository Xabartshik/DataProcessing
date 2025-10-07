# ЛАБОРАТОРНАЯ РАБОТА № 6. ВАРИАНТ 14
# Кластеризация методом k-средних (k-means)
# Условие (варианты №11–15): 6-мерная N(0,1), 50 реализаций; разбить на минимальное число кластеров,
#   чтобы любая точка кластера была на расстоянии < 0.3 от центра (среднего) этого кластера.

import numpy as np
import matplotlib.pyplot as plt
import sys
from io import StringIO
import os
from sklearn.cluster import KMeans


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


sys.stdout = Logger(original_stdout, output_buffer)

print("=== ЛАБОРАТОРНАЯ РАБОТА № 6. ВАРИАНТ 14 ===")
print("Тема: K-средних; нормальная 6-мерная выборка; минимальное k при ограничении радиуса 0.3")
print()

# ========== ИСХОДНЫЕ ДАННЫЕ (50 реализаций) ==========
rng = np.random.default_rng(2002)
n_samples = 50
dim = 6
X = rng.normal(loc=0.0, scale=0.1, size=(n_samples, dim)).astype(float)

print("Сформирована выборка:")
print(f" n = {n_samples}, dim = {dim}")
mu = X.mean(axis=0)
std = X.std(axis=0, ddof=1)
print(" Оценки матожиданий по координатам:", ", ".join(f"{v:.3f}" for v in mu))
print(" Оценки СКО по координатам:", ", ".join(f"{v:.3f}" for v in std))
print()


# ========== РЕАЛИЗАЦИЯ K-MEANS ВРУЧНУЮ (БЕЗ ВНЕШНИХ ЗАВИСИМОСТЕЙ) ==========
def kmeans_numpy(X, k, n_init=10, max_iter=300, tol=1e-4, rng=None):
    """
    Алгоритм k-средних (Lloyd's algorithm) с несколькими случайными инициализациями:

    1. Инициализация: выбираем k случайных точек из выборки как начальные центры кластеров.
    2. E-шаг (Expectation): назначаем каждую точку в ближайший кластер по евклидову расстоянию.
    3. M-шаг (Maximization): пересчитываем центры кластеров как среднее арифметическое точек в каждом кластере.
    4. Повторяем E- и M-шаги, пока изменение центров не станет меньше порога tol или не достигнем max_iter итераций.
    5. Из n_init запусков выбираем решение с минимальной инерцией (сумма квадратов расстояний точек до их центров).

    Возвращает: centers (k, d), labels (n,), inertia (сумма квадратов расстояний до центров)
    """
    n, d = X.shape
    best_inertia = np.inf
    best_centers = None
    best_labels = None
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(n_init):
        # Шаг 1: Инициализация — выбираем k случайных точек как начальные центры
        idx = rng.choice(n, size=k, replace=False)
        centers = X[idx].copy()

        for _it in range(max_iter):
            # Шаг 2 (E-шаг): вычисляем расстояния от каждой точки до каждого центра
            # dists_sq: (n, k) — квадраты евклидовых расстояний, евклидовы метрики
            dists_sq = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            # Назначаем каждую точку в кластер с ближайшим центром
            labels = np.argmin(dists_sq, axis=1)

            # Шаг 3 (M-шаг): пересчитываем центры как средние координаты точек в кластере
            new_centers = centers.copy()
            for j in range(k):
                mask = (labels == j)
                if np.any(mask):
                    # Новый центр — среднее арифметическое всех точек кластера j
                    new_centers[j] = X[mask].mean(axis=0)
                else:
                    # Пустой кластер: переинициализируем случайной точкой из выборки
                    new_centers[j] = X[rng.integers(0, n)]

            # Шаг 4: проверяем критерий остановки (изменение центров меньше tol)
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < tol:
                break

        # Шаг 5: вычисляем итоговую инерцию для текущего запуска
        dists_sq = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dists_sq, axis=1)
        inertia = float(np.take_along_axis(dists_sq, labels[:, None], axis=1).sum())

        # Сохраняем лучшее решение (с минимальной инерцией)
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()
            best_labels = labels.copy()

    return best_centers, best_labels, best_inertia


def cluster_metrics(X, labels, centers):
    """
    Для каждого кластера возвращает словарь:
      size, max_radius (max ||x - c||_2), mean_sqdist, std_radius (СКО расстояний), indices
    """
    k = centers.shape[0]
    metrics = []
    for j in range(k):
        idx = np.where(labels == j)[0]
        if len(idx) == 0:
            metrics.append({
                "size": 0, "max_radius": np.nan, "mean_sqdist": np.nan, "std_radius": np.nan, "indices": idx
            })
            continue
        Xj = X[idx]
        diffs = Xj - centers[j]
        dists = np.linalg.norm(diffs, axis=1)
        max_r = float(dists.max())
        mean_sq = float((dists ** 2).mean())
        std_r = float(dists.std(ddof=1)) if len(dists) > 1 else 0.0
        metrics.append({
            "size": int(len(idx)),
            "max_radius": max_r,
            "mean_sqdist": mean_sq,
            "std_radius": std_r,
            "indices": idx
        })
    return metrics


# ========== ПОИСК МИНИМАЛЬНОГО K ПОД ОГРАНИЧЕНИЕ РАДИУСА < 0.3 (МОЯ РЕАЛИЗАЦИЯ) ==========
radius_threshold = 0.3
print("=== Поиск минимального числа кластеров при ограничении радиуса < 0.3 (kmeans_numpy) ===")
best_solution = None

for k in range(1, n_samples + 1):
    centers, labels, inertia = kmeans_numpy(X, k, n_init=20, max_iter=300, tol=1e-4, rng=rng)
    mets = cluster_metrics(X, labels, centers)
    max_radii = [m["max_radius"] for m in mets if not np.isnan(m["max_radius"])]
    ok = all((r < radius_threshold) for r in max_radii) if len(max_radii) == k else False

    print(f"k = {k:2d}: инерция = {inertia:.4f}; макс. радиусы по кластерам = " +
          (", ".join(f"{r:.3f}" for r in max_radii) if len(max_radii) > 0 else "—") +
          f"; условие выполнено: {'ДА' if ok else 'нет'}")

    if ok:
        best_solution = (k, centers, labels, mets, inertia)
        break

print()
if best_solution is None:
    print("Не удалось удовлетворить ограничение радиуса для k ≤ n_samples — увеличьте порог или проверьте данные")
    sys.stdout = original_stdout
    with open("lab6_output.txt", "w", encoding="utf-8") as f:
        f.write(output_buffer.getvalue())
    print("Готово: полный консольный вывод записан в lab6_output.txt")
    raise SystemExit(0)

k_star, centers, labels, mets, inertia = best_solution
print(f"Минимальное k, удовлетворяющее радиусу < {radius_threshold}: k* = {k_star}")
print()

# ========== СТАТИСТИКА ПО КЛАСТЕРАМ (СОБСТВЕННАЯ РЕАЛИЗАЦИЯ) ==========
print("=== Сводные метрики по кластерам (kmeans_numpy) ===")
for j, m in enumerate(mets):
    if m["size"] == 0:
        print(f" Кластер {j}: пустой")
        continue
    indices_str = ", ".join(str(i) for i in m["indices"])
    print(
        f" Кластер {j}: size={m['size']}, max_radius={m['max_radius']:.4f}, "
        f"mean_sqdist={m['mean_sqdist']:.5f}, std_radius={m['std_radius']:.5f}"
    )
    print(f"   Точки (индексы): {indices_str}")
print()

# ========== ПРОВЕРКА ОГРАНИЧЕНИЯ ==========
all_ok = all((m["size"] > 0 and m["max_radius"] < radius_threshold) for m in mets)
print(f"Проверка условия для всех кластеров (max_radius < {radius_threshold}): {'ВЫПОЛНЕНО' if all_ok else 'НАРУШЕНО'}")
print()

# ========== КЛАСТЕРИЗАЦИЯ С ИСПОЛЬЗОВАНИЕМ SKLEARN ==========
print("=== Поиск минимального числа кластеров при ограничении радиуса < 0.3 (sklearn.KMeans) ===")
best_solution_sklearn = None

for k in range(1, n_samples + 1):
    # Используем sklearn KMeans с аналогичными параметрами
    kmeans_sk = KMeans(n_clusters=k, n_init=20, max_iter=300, tol=1e-4, random_state=42, algorithm='lloyd')
    labels_sk = kmeans_sk.fit_predict(X)
    centers_sk = kmeans_sk.cluster_centers_
    inertia_sk = kmeans_sk.inertia_

    mets_sk = cluster_metrics(X, labels_sk, centers_sk)
    max_radii_sk = [m["max_radius"] for m in mets_sk if not np.isnan(m["max_radius"])]
    ok_sk = all((r < radius_threshold) for r in max_radii_sk) if len(max_radii_sk) == k else False

    print(f"k = {k:2d}: инерция = {inertia_sk:.4f}; макс. радиусы по кластерам = " +
          (", ".join(f"{r:.3f}" for r in max_radii_sk) if len(max_radii_sk) > 0 else "—") +
          f"; условие выполнено: {'ДА' if ok_sk else 'нет'}")

    if ok_sk:
        best_solution_sklearn = (k, centers_sk, labels_sk, mets_sk, inertia_sk)
        break

print()
if best_solution_sklearn is None:
    print("Sklearn: не удалось удовлетворить ограничение радиуса")
else:
    k_star_sk, centers_sk, labels_sk, mets_sk, inertia_sk = best_solution_sklearn
    print(f"Минимальное k (sklearn): k* = {k_star_sk}")
    print()

    print("=== Сводные метрики по кластерам (sklearn.KMeans) ===")
    for j, m in enumerate(mets_sk):
        if m["size"] == 0:
            print(f" Кластер {j}: пустой")
            continue
        indices_str = ", ".join(str(i) for i in m["indices"])
        print(
            f" Кластер {j}: size={m['size']}, max_radius={m['max_radius']:.4f}, "
            f"mean_sqdist={m['mean_sqdist']:.5f}, std_radius={m['std_radius']:.5f}"
        )
        print(f"   Точки (индексы): {indices_str}")
    print()

    all_ok_sk = all((m["size"] > 0 and m["max_radius"] < radius_threshold) for m in mets_sk)
    print(f"Проверка условия (sklearn): {'ВЫПОЛНЕНО' if all_ok_sk else 'НАРУШЕНО'}")
    print()


print("=== Визуализация в 2D (PCA, sklearn) ===")
os.makedirs("figures", exist_ok=True)

from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)  # 2 главные компоненты для 2D
X2d = pca.fit_transform(X)                 # обучаем PCA и проектируем точки
centers2d = pca.transform(centers)         # проектируем центры кластеров

# График для собственной реализации kmeans_numpy
plt.figure(figsize=(11, 6))
palette = plt.get_cmap("tab10", k_star)
colors = palette(np.linspace(0, 1, k_star))
for j in range(k_star):
    idx = np.where(labels == j)[0]
    if len(idx) == 0:
        continue
    plt.scatter(X2d[idx, 0], X2d[idx, 1], s=45, alpha=0.8, color=colors[j],
                label=f"Кластер {j} (n={len(idx)})")
plt.scatter(centers2d[:, 0], centers2d[:, 1], s=180, c="k", marker="X",
            label="Центры (PCA-проекция)")
plt.title(f"ЛР6 Вариант 14: kmeans_numpy (PCA sklearn), k*={k_star}, порог радиуса < {radius_threshold}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, alpha=0.25)
plt.legend(loc="best", ncols=2)
fig_path_numpy = os.path.join("figures", "lab6_variant14_kmeans_numpy.png")
plt.tight_layout()
plt.savefig(fig_path_numpy, dpi=300, bbox_inches="tight")
plt.show()
print(f"График (kmeans_numpy, PCA sklearn) сохранен: {fig_path_numpy}")

# График для реализации sklearn.KMeans (если она нашлась)
if 'best_solution_sklearn' in globals() and best_solution_sklearn is not None:
    # Проекция центров sklearn-кластеризации тем же обученным PCA
    centers2d_sk = pca.transform(centers_sk)

    plt.figure(figsize=(11, 6))
    palette_sk = plt.get_cmap("tab10", k_star_sk)
    colors_sk = palette_sk(np.linspace(0, 1, k_star_sk))
    for j in range(k_star_sk):
        idx = np.where(labels_sk == j)[0]
        if len(idx) == 0:
            continue
        plt.scatter(X2d[idx, 0], X2d[idx, 1], s=45, alpha=0.8, color=colors_sk[j],
                    label=f"Кластер {j} (n={len(idx)})")
    plt.scatter(centers2d_sk[:, 0], centers2d_sk[:, 1], s=180, c="k", marker="X",
                label="Центры (PCA-проекция)")
    plt.title(f"ЛР6 Вариант 14: sklearn.KMeans (PCA sklearn), k*={k_star_sk}, порог радиуса < {radius_threshold}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", ncols=2)
    fig_path_sklearn = os.path.join("figures", "lab6_variant14_kmeans_sklearn.png")
    plt.tight_layout()
    plt.savefig(fig_path_sklearn, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"График (sklearn.KMeans, PCA sklearn) сохранен: {fig_path_sklearn}")

print()

# ========== ИТОГОВАЯ СВОДКА ==========
print("=== ИТОГОВАЯ СВОДКА ===")
print(f"Собственная реализация: k* = {k_star}, инерция = {inertia:.4f}")
if best_solution_sklearn is not None:
    print(f"Sklearn реализация: k* = {k_star_sk}, инерция = {inertia_sk:.4f}")
print()

# ========== СОХРАНЕНИЕ ПОЛНОГО ВЫВОДА В ФАЙЛ ==========
sys.stdout = original_stdout
with open("lab6_output.txt", "w", encoding="utf-8") as f:
    f.write(output_buffer.getvalue())
print("Готово: графики сохранены в figures/, полный консольный вывод записан в lab6_output.txt")
