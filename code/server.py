from fastapi import FastAPI
import numpy as np

from structural_network_engine_v1 import detect_network, analyze_full_structure

app = FastAPI(
    title="Structural Network Engine API",
    version="1.0.0",
    description="API for structural network extraction from 3D point clouds."
)


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    return obj


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

    return to_jsonable({
        "metrics": result.get("metrics", {}),
        "backbone_points": result.get("backbone_points", []),
        "trunk_points": result.get("trunk", {}).get("points", []),
        "topology": result.get("topology", {}),
    })


@app.post("/analyze_full")
def analyze_full(data: dict):
    pts = np.array(data["points"], dtype=float)

    result = analyze_full_structure(
        points=pts,
        mode="v47_compact"
    )

    return to_jsonable(result)
