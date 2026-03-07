from fastapi import FastAPI
import numpy as np

from structural_network_engine_v1 import detect_network

app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok", "service": "Structural Network Engine API"}


@app.post("/analyze")
def analyze(data: dict):
    pts = np.array(data["points"], dtype=float)

    result = detect_network(points=pts, mode="v47_compact")

    return {
        "metrics": result["metrics"],
        "backbone_points": result["backbone_points"].tolist(),
        "trunk_points": result["trunk"]["points"].tolist(),
    }
