from fastapi import FastAPI
import numpy as np

from structural_network_engine_v1 import detect_network

app = FastAPI(
    title="Structural Network Engine API",
    version="1.0.0",
    description="API for structural network extraction from 3D point clouds."
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Structural Network Engine API"
    }


@app.post("/analyze")
def analyze(data: dict):
    pts = np.array(data["points"], dtype=float)

    result = detect_network(points=pts, mode="v47_compact")

    return {
        "metrics": result["metrics"],
        "backbone_points": result["backbone_points"].tolist(),
        "trunk_points": result["trunk"]["points"].tolist(),
        "topology": result["topology"],
    }


@app.post("/analyze_full")
def analyze_full(data: dict):
    pts = np.array(data["points"], dtype=float)

    result = detect_network(points=pts, mode="v47_compact")

    edge_table = result.get("edge_table", [])
    topology = result.get("topology", {})
    trunk = result.get("trunk", {})
    metrics = result.get("metrics", {})

    topological_edges = topology.get("topological_edges", [])
    junctions = topology.get("junctions_global", [])
    endpoints = topology.get("endpoints_global", [])

    branch_lengths = [float(e["length"]) for e in topological_edges if "length" in e]
    avg_branch_length = float(np.mean(branch_lengths)) if branch_lengths else 0.0
    max_branch_length = float(np.max(branch_lengths)) if branch_lengths else 0.0

    hub_nodes = junctions[:20]
    critical_nodes = junctions[:20]

    anomaly_flags = {
        "fragmented_network": bool(metrics.get("components_after_reconnect", 0) > 1),
        "low_confidence_edges": bool(metrics.get("mean_edge_confidence", 0.0) < 0.55),
        "too_many_endpoints": bool(len(endpoints) > max(10, len(junctions) * 3)),
        "low_trunk_straightness": bool(metrics.get("trunk_straightness", 0.0) < 0.25),
    }

    structural_signature = {
        "input_nodes": int(metrics.get("input_nodes", 0)),
        "backbone_nodes": int(metrics.get("backbone_nodes", 0)),
        "topological_nodes": int(metrics.get("topological_nodes", 0)),
        "topological_edges": int(metrics.get("topological_edges", 0)),
        "junction_count": int(len(junctions)),
        "endpoint_count": int(len(endpoints)),
        "trunk_nodes": int(metrics.get("trunk_nodes", 0)),
        "trunk_length": float(metrics.get("trunk_length", 0.0)),
        "trunk_straightness": float(metrics.get("trunk_straightness", 0.0)),
        "mean_edge_confidence": float(metrics.get("mean_edge_confidence", 0.0)),
        "avg_branch_length": float(avg_branch_length),
        "max_branch_length": float(max_branch_length),
    }

    advanced_metrics = {
        "hub_count": int(len(junctions)),
        "endpoint_count": int(len(endpoints)),
        "avg_branch_length": float(avg_branch_length),
        "max_branch_length": float(max_branch_length),
        "network_fragmentation": int(metrics.get("components_after_reconnect", 0)),
        "largest_component_size": int(metrics.get("largest_component", 0)),
    }

    return {
        "status": "ok",
        "mode": result.get("mode", "v47_compact"),
        "metrics": metrics,
        "advanced_metrics": advanced_metrics,
        "anomaly_flags": anomaly_flags,
        "structural_signature": structural_signature,
        "critical_nodes": critical_nodes,
        "hub_nodes": hub_nodes,
        "backbone_points": result["backbone_points"].tolist(),
        "backbone_source_point_ids": result.get("backbone_source_point_ids", []),
        "trunk": {
            "path_ids_local": trunk.get("path_ids_local", []),
            "path_ids_global": trunk.get("path_ids_global", []),
            "points": trunk.get("points", np.empty((0, 3))).tolist(),
            "nodes": trunk.get("nodes", 0),
            "length": trunk.get("length", 0.0),
            "endpoint_distance": trunk.get("endpoint_distance", 0.0),
            "straightness": trunk.get("straightness", 0.0),
        },
        "topology": topology,
        "edge_table": edge_table,
    }
