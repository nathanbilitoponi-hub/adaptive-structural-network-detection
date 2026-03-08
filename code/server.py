from fastapi import FastAPI
import numpy as np

from structural_network_engine_v1 import detect_network, analyze_full_structure

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

    result = analyze_full_structure(
        points=pts,
        mode="v47_compact"
    )

    return result
