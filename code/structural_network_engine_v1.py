import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree, shortest_path
from sklearn.decomposition import PCA


def _safe_unit(v):
    n = np.linalg.norm(v)
    if n == 0:
        return None
    return v / n


def _build_sparse(n, edges):
    rows = []
    cols = []
    vals = []

    for a, b, w in edges:
        if a == b:
            continue

        rows.append(a)
        cols.append(b)
        vals.append(w)

        rows.append(b)
        cols.append(a)
        vals.append(w)

    return csr_matrix((vals, (rows, cols)), shape=(n, n))


def detect_network(points=None, mode="v47_compact", config=None):

    pts = np.asarray(points, float)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be Nx3")

    N = len(pts)

    if N < 3:
        return {
            "metrics": {"input_nodes": int(N)},
            "backbone_points": [],
            "trunk": {"points": []},
        }

    tree = cKDTree(pts)

    k = min(8, N)

    d, idx = tree.query(pts, k=k)

    base_scale = np.median(d[:, 1])

    edges = []

    for i in range(N):

        for j in idx[i][1:]:

            j = int(j)

            if i == j:
                continue

            dist = np.linalg.norm(pts[i] - pts[j])

            if dist < base_scale * 2.5:
                edges.append((i, j, dist))

    G = _build_sparse(N, edges)

    ncomp, labels = connected_components(G)

    mst = minimum_spanning_tree(G)

    mst = mst + mst.T

    rr, cc = mst.nonzero()

    neighbors = {i: [] for i in range(N)}

    for a, b in zip(rr, cc):
        neighbors[a].append(b)
        neighbors[b].append(a)

    endpoints = []

    for i in range(N):
        if len(neighbors[i]) == 1:
            endpoints.append(i)

    if len(endpoints) < 2:
        endpoints = list(range(min(2, N)))

    D, pred = shortest_path(mst, directed=False, return_predecessors=True)

    s = endpoints[0]
    t = endpoints[-1]

    path = [t]

    cur = t

    while cur != s and cur != -9999:
        cur = pred[s, cur]
        if cur == -9999:
            break
        path.append(cur)

    path = path[::-1]

    trunk_points = pts[path]

    backbone_ids = []

    pca_k = min(10, N)

    for i in range(N):

        _, ids = tree.query(pts[i], k=pca_k)

        neigh = pts[np.atleast_1d(ids)]

        if len(neigh) < 3:
            continue

        try:

            pca = PCA(n_components=3)

            pca.fit(neigh)

            eig = pca.explained_variance_

            filamentarity = eig[0] / (eig[1] + 1e-12)

            if filamentarity > 3.0:
                backbone_ids.append(i)

        except:
            pass

    backbone_points = pts[backbone_ids]

    metrics = {
        "input_nodes": int(N),
        "filament_nodes": int(len(backbone_ids)),
        "graph_edges": int(len(edges)),
        "components": int(ncomp),
        "trunk_nodes": int(len(trunk_points)),
    }

    return {
        "metrics": metrics,
        "backbone_points": backbone_points,
        "trunk": {"points": trunk_points},
    }

  
