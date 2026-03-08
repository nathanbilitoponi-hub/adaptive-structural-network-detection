import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree, shortest_path
from sklearn.decomposition import PCA


DEFAULT_CONFIG = {
    "local_k": 8,
    "filament_degree_min": 3,
    "pca_k": 10,
    "filamentarity_threshold": 3.0,
    "direction_k": 8,
    "graph_k": 10,
    "edge_cut_scale": 2.6,
    "collinearity_min": 0.45,
    "base_score_floor": 0.15,
    "reconnect_candidate_k": 5,
    "reconnect_limit_scale": 3.0,
}


def _cfg(user_config=None):
    c = dict(DEFAULT_CONFIG)
    if user_config:
        c.update(user_config)
    return c


def _safe_unit(v):
    n = np.linalg.norm(v)
    return None if n == 0 else v / n


def _build_sparse(n, edges):
    rows, cols, vals = [], [], []
    for a, b, w in edges:
        if a == b:
            continue
        rows += [int(a), int(b)]
        cols += [int(b), int(a)]
        vals += [float(w), float(w)]
    return csr_matrix((vals, (rows, cols)), shape=(n, n))


def _largest_component_graph(G):
    if G.shape[0] == 0:
        return {
            "graph": csr_matrix((0, 0)),
            "ids": np.array([], dtype=int),
            "ncomp": 0,
            "largest_size": 0,
        }

    ncomp, labels = connected_components(G, directed=False)
    sizes = np.bincount(labels)
    lid = int(np.argmax(sizes))
    ids = np.where(labels == lid)[0]

    remap = {old: i for i, old in enumerate(ids)}
    coo = G.tocoo()

    rows, cols, vals = [], [], []
    for a, b, w in zip(coo.row, coo.col, coo.data):
        if a in remap and b in remap:
            rows.append(remap[a])
            cols.append(remap[b])
            vals.append(float(w))

    return {
        "graph": csr_matrix((vals, (rows, cols)), shape=(len(ids), len(ids))),
        "ids": ids,
        "ncomp": int(ncomp),
        "largest_size": int(sizes.max()),
    }


def _local_dir(points, tree, i, k):
    if len(points) < 3:
        return None

    kk = min(max(3, int(k)), len(points))
    _, ids = tree.query(points[i], k=kk)
    neigh = points[np.atleast_1d(ids)]

    if len(neigh) < 3:
        return None

    X = neigh - neigh.mean(axis=0, keepdims=True)
    C = np.cov(X.T)
    w, v = np.linalg.eigh(C)
    vec = v[:, np.argsort(w)[::-1][0]]
    return _safe_unit(vec)


def _path_length(pts):
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _mst_longest_path(points, mst_graph, global_ids=None):
    n = len(points)
    if n == 0:
        return {
            "path_ids_local": [],
            "path_ids_global": [],
            "points": np.empty((0, 3)),
            "nodes": 0,
            "length": 0.0,
            "endpoint_distance": 0.0,
            "straightness": 0.0,
        }

    if n == 1:
        gid = [int(global_ids[0])] if global_ids is not None else [0]
        return {
            "path_ids_local": [0],
            "path_ids_global": gid,
            "points": points[[0]],
            "nodes": 1,
            "length": 0.0,
            "endpoint_distance": 0.0,
            "straightness": 0.0,
        }

    D, pred = shortest_path(mst_graph, directed=False, return_predecessors=True)
    D[~np.isfinite(D)] = -np.inf
    np.fill_diagonal(D, -np.inf)

    s, t = np.unravel_index(np.argmax(D), D.shape)

    path = [int(t)]
    cur = int(t)
    while cur != int(s) and cur != -9999:
        cur = int(pred[s, cur])
        if cur == -9999:
            break
        path.append(cur)

    path = path[::-1]
    trunk = points[path] if len(path) else points[:1]

    L = _path_length(trunk)
    E = float(np.linalg.norm(trunk[-1] - trunk[0])) if len(trunk) >= 2 else 0.0
    S = E / L if L > 0 else 0.0

    gpath = [int(global_ids[x]) for x in path] if global_ids is not None else [int(x) for x in path]

    return {
        "path_ids_local": [int(x) for x in path],
        "path_ids_global": gpath,
        "points": trunk,
        "nodes": int(len(trunk)),
        "length": float(L),
        "endpoint_distance": float(E),
        "straightness": float(S),
    }


def _extract_topology(points, mst_graph, global_ids=None, fallback_nodes=20):
    n = len(points)
    neighbors = {i: set() for i in range(n)}

    rr, cc = mst_graph.nonzero()
    for a, b in zip(rr, cc):
        if a != b:
            neighbors[int(a)].add(int(b))

    endpoints, junctions = [], []
    for i in range(n):
        deg = len(neighbors[i])
        if deg == 1:
            endpoints.append(i)
        elif deg >= 3:
            junctions.append(i)

    critical = set(endpoints + junctions)
    if len(critical) < 2:
        critical = set(range(min(fallback_nodes, n)))

    visited = set()
    paths = []

    for s in critical:
        for nxt0 in neighbors[s]:
            e0 = tuple(sorted((s, nxt0)))
            if e0 in visited:
                continue

            path = [s, nxt0]
            prev, cur = s, nxt0
            visited.add(e0)

            while True:
                if cur in critical and cur != s:
                    break

                nxts = [x for x in neighbors[cur] if x != prev]
                if not nxts:
                    break

                nxt = nxts[0]
                ek = tuple(sorted((cur, nxt)))
                if ek in visited:
                    break

                path.append(nxt)
                visited.add(ek)
                prev, cur = cur, nxt

            if len(path) >= 2 and path[-1] in critical and path[-1] != path[0]:
                paths.append(path)

    best = {}
    for p in paths:
        key = tuple(sorted((p[0], p[-1])))
        L = 0.0
        for i in range(len(p) - 1):
            L += np.linalg.norm(points[p[i + 1]] - points[p[i]])
        if key not in best or L < best[key]["length"]:
            best[key] = {"path": p, "length": float(L)}

    topo_paths = [v["path"] for v in best.values()]
    topo_nodes_local = sorted(set([p[0] for p in topo_paths] + [p[-1] for p in topo_paths]))

    def gid(i):
        return int(global_ids[i]) if global_ids is not None else int(i)

    topo_edges = []
    for p in topo_paths:
        L = 0.0
        for i in range(len(p) - 1):
            L += np.linalg.norm(points[p[i + 1]] - points[p[i]])

        topo_edges.append({
            "u_local": int(p[0]),
            "v_local": int(p[-1]),
            "u_global": gid(p[0]),
            "v_global": gid(p[-1]),
            "nodes": int(len(p)),
            "length": float(L),
            "path_local": [int(x) for x in p],
            "path_global": [gid(x) for x in p],
        })

    return {
        "topological_node_ids_local": [int(x) for x in topo_nodes_local],
        "topological_node_ids_global": [gid(x) for x in topo_nodes_local],
        "topological_edges": topo_edges,
        "endpoints_local": [int(x) for x in endpoints],
        "junctions_local": [int(x) for x in junctions],
        "endpoints_global": [gid(x) for x in endpoints],
        "junctions_global": [gid(x) for x in junctions],
    }


def _empty_result(points, reason, config):
    pts = np.asarray(points, float)
    return {
        "mode": "v47_compact",
        "config": config,
        "status": "degenerate",
        "reason": str(reason),
        "input_points": pts,
        "filament_node_ids": [],
        "backbone_source_point_ids": [],
        "backbone_points": np.empty((0, 3)),
        "edge_table": [],
        "dense_graph": csr_matrix((0, 0)),
        "largest_component_ids_local": [],
        "largest_component_ids_global": [],
        "mst_graph_lcc": csr_matrix((0, 0)),
        "trunk": {
            "path_ids_local": [],
            "path_ids_global": [],
            "points": np.empty((0, 3)),
            "nodes": 0,
            "length": 0.0,
            "endpoint_distance": 0.0,
            "straightness": 0.0,
        },
        "topology": {
            "topological_node_ids_local": [],
            "topological_node_ids_global": [],
            "topological_edges": [],
            "endpoints_local": [],
            "junctions_local": [],
            "endpoints_global": [],
            "junctions_global": [],
        },
        "metrics": {
            "input_nodes": int(len(pts)),
            "filament_nodes": 0,
            "backbone_nodes": 0,
            "graph_edges": 0,
            "components_before_reconnect": 0,
            "components_after_reconnect": 0,
            "largest_component": 0,
            "topological_nodes": 0,
            "topological_edges": 0,
            "trunk_nodes": 0,
            "trunk_length": 0.0,
            "trunk_straightness": 0.0,
            "reconnect_edges": 0,
            "global_connected_after_reconnect": False,
            "base_scale": 0.0,
            "mean_backbone_degree": 0.0,
            "mean_edge_confidence": 0.0,
        }
    }


def detect_network_compact(points, config=None):
    cfg = _cfg(config)
    pts = np.asarray(points, float)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(pts) < 3:
        return _empty_result(pts, "too_few_input_points", cfg)

    tree = cKDTree(pts)
    k0 = min(max(2, cfg["local_k"]), len(pts))
    d, idx = tree.query(pts, k=k0)

    if np.atleast_2d(d).shape[1] < 2:
        return _empty_result(pts, "local_graph_failed", cfg)

    edge_scale = 1.5 * np.median(np.atleast_2d(d)[:, 1])

    undirected_edges = set()
    for i in range(len(pts)):
        for j in np.atleast_1d(idx[i])[1:]:
            j = int(j)
            if i == j:
                continue
            dij = np.linalg.norm(pts[i] - pts[j])
            if dij < edge_scale:
                undirected_edges.add(tuple(sorted((int(i), j))))

    degree = np.zeros(len(pts), dtype=int)
    for a, b in undirected_edges:
        degree[a] += 1
        degree[b] += 1

    filament_nodes = np.where(degree >= cfg["filament_degree_min"])[0]
    if len(filament_nodes) < 3:
        return _empty_result(pts, "no_filament_nodes", cfg)

    fil_pts = pts[filament_nodes]
    tree_fil = cKDTree(fil_pts)
    pca_k = min(max(3, cfg["pca_k"]), len(fil_pts))
    backbone_ids_local = []

    for i in range(len(fil_pts)):
        _, ids = tree_fil.query(fil_pts[i], k=pca_k)
        neigh = fil_pts[np.atleast_1d(ids)]
        if len(neigh) < 3:
            continue

        try:
            pca = PCA(n_components=3)
            pca.fit(neigh)
            eig = pca.explained_variance_
            filamentarity = eig[0] / (eig[1] + 1e-12)
            if filamentarity > cfg["filamentarity_threshold"]:
                backbone_ids_local.append(i)
        except Exception:
            pass

    backbone_ids_local = np.array(sorted(set(backbone_ids_local)), dtype=int)
    if len(backbone_ids_local) == 0:
        return _empty_result(pts, "no_backbone_nodes", cfg)

    backbone_source_point_ids = filament_nodes[backbone_ids_local]
    backbone = fil_pts[backbone_ids_local]

    if len(backbone) == 1:
        out = _empty_result(pts, "single_backbone_node", cfg)
        out["status"] = "ok"
        out["filament_node_ids"] = filament_nodes.tolist()
        out["backbone_source_point_ids"] = backbone_source_point_ids.tolist()
        out["backbone_points"] = backbone
        out["largest_component_ids_local"] = [0]
        out["largest_component_ids_global"] = [int(backbone_source_point_ids[0])]
        out["trunk"] = {
            "path_ids_local": [0],
            "path_ids_global": [int(backbone_source_point_ids[0])],
            "points": backbone[[0]],
            "nodes": 1,
            "length": 0.0,
            "endpoint_distance": 0.0,
            "straightness": 0.0,
        }
        out["metrics"]["input_nodes"] = int(len(pts))
        out["metrics"]["filament_nodes"] = int(len(filament_nodes))
        out["metrics"]["backbone_nodes"] = 1
        out["metrics"]["largest_component"] = 1
        out["metrics"]["components_before_reconnect"] = 1
        out["metrics"]["components_after_reconnect"] = 1
        out["metrics"]["global_connected_after_reconnect"] = True
        return out

    tree_back = cKDTree(backbone)
    dirs = np.array([_local_dir(backbone, tree_back, i, cfg["direction_k"]) for i in range(len(backbone))], dtype=object)

    kg = min(max(2, cfg["graph_k"]), len(backbone))
    d1, i1 = tree_back.query(backbone, k=kg)
    d1 = np.atleast_2d(d1)
    i1 = np.atleast_2d(i1)

    if d1.shape[1] < 2:
        return _empty_result(pts, "backbone_graph_failed", cfg)

    base_scale = float(np.median(d1[:, 1]))
    edge_cut = cfg["edge_cut_scale"] * max(base_scale, 1e-12)

    edge_table = []
    for a in range(len(backbone)):
        da = dirs[a]

        for b in np.atleast_1d(i1[a])[1:]:
            b = int(b)
            if b <= a:
                continue

            dist_ab = np.linalg.norm(backbone[a] - backbone[b])
            if dist_ab > edge_cut:
                continue

            db = dirs[b]

            if da is not None and db is not None:
                col = abs(np.dot(da, db))
            else:
                col = 0.5

            if col < cfg["collinearity_min"]:
                continue

            diff = backbone[b] - backbone[a]
            nd = np.linalg.norm(diff)
            if nd == 0:
                continue
            u = diff / nd

            if da is not None and db is not None:
                align = 0.5 * (abs(np.dot(da, u)) + abs(np.dot(db, u)))
            else:
                align = 0.5

            dist_score = float(np.exp(-dist_ab / max(base_scale, 1e-12)))
            score = 0.45 * dist_score + 0.30 * col + 0.25 * align
            cost = dist_ab / max(score, cfg["base_score_floor"])
            confidence = 0.4 * col + 0.4 * align + 0.2 * dist_score

            edge_table.append({
                "u": int(a),
                "v": int(b),
                "cost": float(cost),
                "confidence": float(confidence),
            })

    if not edge_table:
        for i in range(len(backbone)):
            if kg >= 2:
                j = int(np.atleast_1d(i1[i])[1])
                dij = np.linalg.norm(backbone[i] - backbone[j])
                edge_table.append({
                    "u": int(i),
                    "v": int(j),
                    "cost": float(dij),
                    "confidence": 0.1,
                })

    best_edges = {}
    for e in edge_table:
        key = tuple(sorted((e["u"], e["v"])))
        if key not in best_edges or e["cost"] < best_edges[key]["cost"]:
            best_edges[key] = e
    edge_table = list(best_edges.values())

    G0 = _build_sparse(len(backbone), [(e["u"], e["v"], e["cost"]) for e in edge_table])
    ncomp0, lab0 = connected_components(G0, directed=False)

    reconnect_edges = []
    comp_ids = np.unique(lab0)
    comp_members = {int(cid): np.where(lab0 == cid)[0] for cid in comp_ids}

    if len(comp_ids) > 1:
        comp_centers = np.array([backbone[comp_members[int(cid)]].mean(axis=0) for cid in comp_ids])
        center_tree = cKDTree(comp_centers)

        kc = min(max(2, cfg["reconnect_candidate_k"]), len(comp_centers))
        _, ic = center_tree.query(comp_centers, k=kc)
        ic = np.atleast_2d(ic)

        reconnect_limit = cfg["reconnect_limit_scale"] * edge_cut
        comp_candidates = {}

        for ii, cid_a in enumerate(comp_ids):
            ids_a = comp_members[int(cid_a)]

            for p in range(1, ic.shape[1]):
                jj = int(ic[ii, p])
                cid_b = int(comp_ids[jj])
                if cid_a == cid_b:
                    continue

                ids_b = comp_members[cid_b]
                best = None
                best_cost = np.inf

                for a in ids_a:
                    pa = backbone[a]
                    da = dirs[a]

                    diff = backbone[ids_b] - pa
                    dvec = np.linalg.norm(diff, axis=1)
                    if len(dvec) == 0:
                        continue

                    jloc = int(np.argmin(dvec))
                    b = int(ids_b[jloc])
                    dist_ab = float(dvec[jloc])

                    if dist_ab > reconnect_limit:
                        continue

                    db = dirs[b]
                    if da is not None and db is not None:
                        col = abs(np.dot(da, db))
                        u = (backbone[b] - backbone[a]) / max(dist_ab, 1e-12)
                        align = 0.5 * (abs(np.dot(da, u)) + abs(np.dot(db, u)))
                    else:
                        col = 0.0
                        align = 0.0

                    dist_score = float(np.exp(-dist_ab / max(base_scale, 1e-12)))
                    conf = 0.4 * col + 0.4 * align + 0.2 * dist_score
                    cost = dist_ab * (1.20 - 0.20 * col) * (1.15 - 0.15 * align)

                    if cost < best_cost:
                        best_cost = cost
                        best = {
                            "u": int(a),
                            "v": int(b),
                            "cost": float(cost),
                            "confidence": float(conf),
                        }

                if best is not None:
                    key = tuple(sorted((int(cid_a), int(cid_b))))
                    if key not in comp_candidates or best["cost"] < comp_candidates[key]["cost"]:
                        comp_candidates[key] = best

        if comp_candidates:
            comp_id_to_idx = {int(cid): i for i, cid in enumerate(comp_ids)}
            comp_edges = []
            keys_by_pair = {}

            for (cid_a, cid_b), rec in comp_candidates.items():
                ia = comp_id_to_idx[int(cid_a)]
                ib = comp_id_to_idx[int(cid_b)]
                comp_edges.append((ia, ib, float(rec["cost"])))
                keys_by_pair[tuple(sorted((ia, ib)))] = (cid_a, cid_b)

            Gc = _build_sparse(len(comp_ids), comp_edges)
            Tc = minimum_spanning_tree(Gc)
            Tc = Tc + Tc.T
            rr, cc = Tc.nonzero()

            for a, b in zip(rr, cc):
                if a < b:
                    cid_a, cid_b = keys_by_pair[tuple(sorted((int(a), int(b))))]
                    reconnect_edges.append(comp_candidates[tuple(sorted((cid_a, cid_b)))])

    merged = {}
    for e in edge_table + reconnect_edges:
        key = tuple(sorted((e["u"], e["v"])))
        if key not in merged or e["cost"] < merged[key]["cost"]:
            merged[key] = e
        else:
            merged[key]["confidence"] = max(merged[key]["confidence"], e["confidence"])

    edge_table = list(merged.values())
    G_dense = _build_sparse(len(backbone), [(e["u"], e["v"], e["cost"]) for e in edge_table])

    ncomp1, _ = connected_components(G_dense, directed=False)
    lcc = _largest_component_graph(G_dense)

    lcc_local_ids = lcc["ids"]
    lcc_global_ids = backbone_source_point_ids[lcc_local_ids]
    lcc_points = backbone[lcc_local_ids]

    if len(lcc_points) == 0:
        return _empty_result(pts, "empty_lcc", cfg)

    mst = csr_matrix((1, 1)) if len(lcc_points) == 1 else minimum_spanning_tree(lcc["graph"])
    mst = mst + mst.T

    topo = _extract_topology(lcc_points, mst, global_ids=lcc_global_ids)
    trunk = _mst_longest_path(lcc_points, mst, global_ids=lcc_global_ids)

    backbone_degree = np.array(G_dense.getnnz(axis=1)).ravel() if G_dense.shape[0] > 0 else np.array([])
    mean_backbone_degree = float(backbone_degree.mean()) if len(backbone_degree) else 0.0
    mean_edge_confidence = float(np.mean([e["confidence"] for e in edge_table])) if edge_table else 0.0

    return {
        "mode": "v47_compact",
        "config": cfg,
        "status": "ok",
        "reason": "success",
        "input_points": pts,
        "filament_node_ids": filament_nodes.tolist(),
        "backbone_source_point_ids": backbone_source_point_ids.tolist(),
        "backbone_points": backbone,
        "edge_table": edge_table,
        "dense_graph": G_dense,
        "largest_component_ids_local": lcc_local_ids.tolist(),
        "largest_component_ids_global": lcc_global_ids.tolist(),
        "mst_graph_lcc": mst,
        "trunk": trunk,
        "topology": topo,
        "metrics": {
            "input_nodes": int(len(pts)),
            "filament_nodes": int(len(filament_nodes)),
            "backbone_nodes": int(len(backbone)),
            "graph_edges": int(G_dense.nnz // 2),
            "components_before_reconnect": int(ncomp0),
            "components_after_reconnect": int(ncomp1),
            "largest_component": int(lcc["largest_size"]),
            "topological_nodes": int(len(topo["topological_node_ids_global"])),
            "topological_edges": int(len(topo["topological_edges"])),
            "trunk_nodes": int(trunk["nodes"]),
            "trunk_length": float(trunk["length"]),
            "trunk_straightness": float(trunk["straightness"]),
            "reconnect_edges": int(len(reconnect_edges)),
            "global_connected_after_reconnect": bool(ncomp1 == 1),
            "base_scale": float(base_scale),
            "mean_backbone_degree": float(mean_backbone_degree),
            "mean_edge_confidence": float(mean_edge_confidence),
        }
    }


def detect_network(points=None, mode="v47_compact", config=None):
    mode = str(mode).lower()

    if mode in ["v47_compact", "compact", "v47"]:
        if points is None:
            raise ValueError("Provide points")
        return detect_network_compact(points, config=config)

    raise ValueError(f"Unsupported mode: {mode}")


def compute_structural_metrics(result):
    """
    Compute higher-level structural metrics from detect_network output.
    Safe on partial / degenerate outputs.
    """
    metrics = result.get("metrics", {})
    topo = result.get("topology", {})
    trunk = result.get("trunk", {})

    topo_edges = topo.get("topological_edges", [])
    topological_nodes = topo.get("topological_node_ids_global", [])

    edge_lengths = [float(e.get("length", 0.0)) for e in topo_edges]
    total_topology_length = float(np.sum(edge_lengths)) if edge_lengths else 0.0
    average_branch_length = float(np.mean(edge_lengths)) if edge_lengths else 0.0
    max_branch_length = float(np.max(edge_lengths)) if edge_lengths else 0.0

    node_degree = {}
    for e in topo_edges:
        u = int(e["u_global"])
        v = int(e["v_global"])
        node_degree[u] = node_degree.get(u, 0) + 1
        node_degree[v] = node_degree.get(v, 0) + 1

    critical_nodes = [int(n) for n, d in node_degree.items() if d >= 3]
    hub_nodes = sorted(
        [{"node_id": int(n), "degree": int(d)} for n, d in node_degree.items()],
        key=lambda x: x["degree"],
        reverse=True
    )[:10]

    topological_node_count = int(len(topological_nodes))
    topological_edge_count = int(len(topo_edges))

    if topological_node_count > 1:
        network_density = float(
            (2.0 * topological_edge_count) /
            (topological_node_count * (topological_node_count - 1))
        )
    else:
        network_density = 0.0

    branch_lengths_sorted = sorted(edge_lengths, reverse=True)
    branch_length_std = float(np.std(edge_lengths)) if edge_lengths else 0.0
    trunk_length = float(trunk.get("length", 0.0))
    trunk_straightness = float(trunk.get("straightness", 0.0))

    return {
        "topological_node_count": topological_node_count,
        "topological_edge_count": topological_edge_count,
        "total_topology_length": total_topology_length,
        "average_branch_length": average_branch_length,
        "max_branch_length": max_branch_length,
        "branch_length_std": branch_length_std,
        "network_density": network_density,
        "critical_node_count": int(len(critical_nodes)),
        "critical_nodes": critical_nodes[:50],
        "hub_nodes": hub_nodes,
        "largest_branches": branch_lengths_sorted[:10],
        "trunk_length": trunk_length,
        "trunk_straightness": trunk_straightness,
        "global_connected": bool(metrics.get("global_connected_after_reconnect", False)),
    }


def detect_structural_anomalies(result):
    """
    Simple rule-based anomaly detection.
    This is intentionally interpretable and lightweight.
    """
    metrics = result.get("metrics", {})
    topo = result.get("topology", {})
    trunk = result.get("trunk", {})

    anomalies = []

    components_after = int(metrics.get("components_after_reconnect", 0))
    if components_after > 1:
        anomalies.append({
            "type": "fragmented_network",
            "severity": "medium" if components_after <= 3 else "high",
            "value": components_after,
            "message": f"Network remains split into {components_after} components."
        })

    mean_edge_conf = float(metrics.get("mean_edge_confidence", 0.0))
    if mean_edge_conf < 0.45:
        anomalies.append({
            "type": "low_edge_confidence",
            "severity": "medium",
            "value": mean_edge_conf,
            "message": "Average edge confidence is low."
        })

    trunk_straightness = float(trunk.get("straightness", 0.0))
    if trunk_straightness < 0.15 and int(trunk.get("nodes", 0)) > 10:
        anomalies.append({
            "type": "high_trunk_tortuosity",
            "severity": "low",
            "value": trunk_straightness,
            "message": "Trunk is highly tortuous / non-straight."
        })

    topo_edges = topo.get("topological_edges", [])
    edge_lengths = [float(e.get("length", 0.0)) for e in topo_edges]
    if edge_lengths:
        mean_len = float(np.mean(edge_lengths))
        std_len = float(np.std(edge_lengths))
        threshold = mean_len + 2.5 * std_len

        long_edges = []
        for e in topo_edges:
            L = float(e.get("length", 0.0))
            if L > threshold:
                long_edges.append({
                    "u_global": int(e["u_global"]),
                    "v_global": int(e["v_global"]),
                    "length": L
                })

        if long_edges:
            anomalies.append({
                "type": "abnormally_long_branches",
                "severity": "medium",
                "count": len(long_edges),
                "message": "Some branches are significantly longer than average.",
                "examples": long_edges[:10]
            })

    return {
        "anomaly_count": int(len(anomalies)),
        "anomalies": anomalies
    }


def compute_structural_signature(result):
    """
    Compact signature vector describing the extracted structure.
    Useful for comparisons and future ML / indexing.
    """
    metrics = result.get("metrics", {})
    topo = result.get("topology", {})
    trunk = result.get("trunk", {})

    topo_edges = topo.get("topological_edges", [])
    edge_lengths = [float(e.get("length", 0.0)) for e in topo_edges]

    node_degree = {}
    for e in topo_edges:
        u = int(e["u_global"])
        v = int(e["v_global"])
        node_degree[u] = node_degree.get(u, 0) + 1
        node_degree[v] = node_degree.get(v, 0) + 1

    degrees = list(node_degree.values())
    mean_degree = float(np.mean(degrees)) if degrees else 0.0
    max_degree = int(np.max(degrees)) if degrees else 0
    hub_ratio = float(np.mean(np.array(degrees) >= 3)) if degrees else 0.0

    signature = {
        "input_nodes": int(metrics.get("input_nodes", 0)),
        "backbone_nodes": int(metrics.get("backbone_nodes", 0)),
        "graph_edges": int(metrics.get("graph_edges", 0)),
        "topological_nodes": int(metrics.get("topological_nodes", 0)),
        "topological_edges": int(metrics.get("topological_edges", 0)),
        "trunk_nodes": int(trunk.get("nodes", 0)),
        "trunk_length": float(trunk.get("length", 0.0)),
        "trunk_straightness": float(trunk.get("straightness", 0.0)),
        "mean_edge_confidence": float(metrics.get("mean_edge_confidence", 0.0)),
        "mean_degree": mean_degree,
        "max_degree": max_degree,
        "hub_ratio": hub_ratio,
        "mean_branch_length": float(np.mean(edge_lengths)) if edge_lengths else 0.0,
        "branch_length_std": float(np.std(edge_lengths)) if edge_lengths else 0.0,
        "global_connected": int(bool(metrics.get("global_connected_after_reconnect", False))),
    }

    signature_vector = [
        signature["input_nodes"],
        signature["backbone_nodes"],
        signature["graph_edges"],
        signature["topological_nodes"],
        signature["topological_edges"],
        signature["trunk_nodes"],
        signature["trunk_length"],
        signature["trunk_straightness"],
        signature["mean_edge_confidence"],
        signature["mean_degree"],
        signature["max_degree"],
        signature["hub_ratio"],
        signature["mean_branch_length"],
        signature["branch_length_std"],
        signature["global_connected"],
    ]

    return {
        "signature": signature,
        "signature_vector": signature_vector
    }


def analyze_full_structure(points, mode="v47_compact", config=None):
    """
    Full premium analysis wrapper.
    """
    result = detect_network(points=points, mode=mode, config=config)

    return {
        "status": result.get("status", "unknown"),
        "reason": result.get("reason", ""),
        "mode": result.get("mode", mode),
        "metrics_base": result.get("metrics", {}),
        "trunk": result.get("trunk", {}),
        "topology": result.get("topology", {}),
        "structural_metrics": compute_structural_metrics(result),
        "anomaly_detection": detect_structural_anomalies(result),
        "structural_signature": compute_structural_signature(result),
    }
