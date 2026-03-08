import numpy as np


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
