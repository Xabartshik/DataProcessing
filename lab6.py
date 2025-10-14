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


# === Визуализация на сфере (две версии: kmeans_numpy и sklearn.KMeans) ===
print("=== Визуализация на сфере (Plotly): мое kmeans и sklearn.KMeans ===")

import os
os.makedirs("figures", exist_ok=True)

import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.io as pio

# 1) Проекция в 3D и нормировка на единичную сферу
pca3 = PCA(n_components=3, random_state=42)
X3d = pca3.fit_transform(X)

def normalize_rows(A):
    nrm = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.clip(nrm, 1e-12, None)

P = normalize_rows(X3d)  # точки на S^2 (общие для обеих визуализаций)

# Палитра цветов
import plotly.colors as pc
palette = pc.qualitative.Plotly
def color_for(j):
    return palette[j % len(palette)]

# Вспомогательная функция построения опорной сферы как Mesh3d
def make_support_sphere(res_phi=40, res_theta=80):
    phi = np.linspace(0, np.pi, res_phi)
    theta = np.linspace(0, 2*np.pi, res_theta)
    pp, tt = np.meshgrid(phi, theta, indexing="ij")
    xs = (np.sin(pp) * np.cos(tt)).ravel()
    ys = (np.sin(pp) * np.sin(tt)).ravel()
    zs = (np.cos(pp)).ravel()
    return go.Mesh3d(
        x=xs, y=ys, z=zs,
        color="lightgray", opacity=0.15,
        alphahull=0,
        name="sphere"
    )

# Вспомогательная функция построения сферических областей (Вороной) по центрам
def make_spherical_voronoi_patches(C, colors):
    patches = []
    try:
        from scipy.spatial import SphericalVoronoi
        sv = SphericalVoronoi(C, radius=1.0, center=np.array([0.0, 0.0, 0.0]))
        sv.sort_vertices_of_regions()
        for cid, region in enumerate(sv.regions):
            verts = sv.vertices[region]  # (m,3)
            centroid = verts.mean(axis=0)
            centroid /= np.linalg.norm(centroid)
            verts_all = np.vstack([centroid[None, :], verts])
            tri_i, tri_j, tri_k = [], [], []
            for j in range(len(verts)):
                a = 1 + j
                b = 1 + ((j + 1) % len(verts))
                tri_i.append(0); tri_j.append(a); tri_k.append(b)
            patches.append(go.Mesh3d(
                x=verts_all[:,0], y=verts_all[:,1], z=verts_all[:,2],
                i=tri_i, j=tri_j, k=tri_k,
                color=colors[cid % len(colors)], opacity=0.45,
                name=f"cluster {cid} region"
            ))
    except Exception as e:
        print(f"[WARN] SphericalVoronoi не построен: {e}")
    return patches

# Вспомогательная функция scatter-точек по меткам
def make_cluster_scatters(P, labels, k):
    scatters = []
    for j in range(k):
        m = (labels == j)
        if not np.any(m):
            continue
        d = P[m]
        scatters.append(go.Scatter3d(
            x=d[:,0], y=d[:,1], z=d[:,2],
            mode="markers",
            marker=dict(size=4, color=color_for(j)),
            name=f"cluster {j}"
        ))
    return scatters

# 2) Сфера для моей реализации kmeans_numpy
centers3d = pca3.transform(centers)
C = normalize_rows(centers3d)

sphere_numpy = make_support_sphere()
patches_numpy = make_spherical_voronoi_patches(C, [color_for(j) for j in range(k_star)])
scatters_numpy = make_cluster_scatters(P, labels, k_star)

fig_numpy = go.Figure(data=[sphere_numpy] + patches_numpy + scatters_numpy)
fig_numpy.update_layout(
    scene=dict(
        xaxis=dict(showgrid=False, zeroline=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, title=""),
        zaxis=dict(showgrid=False, zeroline=False, title=""),
        aspectmode="data"
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    title=f"ЛР6: кластеры на сфере (моё kmeans, k*={k_star}, порог < {radius_threshold})"
)
sphere_html_numpy = os.path.join("figures", "lab6_sphere_numpy.html")
pio.write_html(fig_numpy, file=sphere_html_numpy, auto_open=False, include_plotlyjs="cdn")
print(f"[OK] Сфера (моё kmeans) сохранена в {sphere_html_numpy}")

# 3) Сфера для sklearn.KMeans (если решение найдено)
if 'best_solution_sklearn' in globals() and best_solution_sklearn is not None:
    centers3d_sk = pca3.transform(centers_sk)
    C_sk = normalize_rows(centers3d_sk)

    sphere_sklearn = make_support_sphere()
    patches_sklearn = make_spherical_voronoi_patches(C_sk, [color_for(j) for j in range(k_star_sk)])
    scatters_sklearn = make_cluster_scatters(P, labels_sk, k_star_sk)

    fig_sklearn = go.Figure(data=[sphere_sklearn] + patches_sklearn + scatters_sklearn)
    fig_sklearn.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False, title=""),
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        title=f"ЛР6: кластеры на сфере (sklearn, k*={k_star_sk}, порог < {radius_threshold})"
    )
    sphere_html_sklearn = os.path.join("figures", "lab6_sphere_sklearn.html")
    pio.write_html(fig_sklearn, file=sphere_html_sklearn, auto_open=False, include_plotlyjs="cdn")
    print(f"[OK] Сфера (sklearn.KMeans) сохранена в {sphere_html_sklearn}")



