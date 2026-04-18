from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

from inference.predict_network import predict_network
from inference.predict_web import predict_web
from inference.predict_windows import predict_windows
from inference.predict_url_lexical import predict_url_raw
from utils.scoring import calculate_security_score

app = FastAPI(title="AI Cyber Defender API")


class URLRawRequest(BaseModel):
    url: str


class FeaturesRequest(BaseModel):
    features: Dict[str, Any]


@app.get("/")
def root():
    return {"message": "AI Cyber Defender API is running"}


@app.post("/predict/url_raw")
def predict_url_raw_endpoint(request: URLRawRequest):
    return predict_url_raw(request.url)


@app.post("/predict/network")
def predict_network_endpoint(request: FeaturesRequest):
    result = predict_network(request.features)
    return {**result, **calculate_security_score(result)}


@app.post("/predict/web")
def predict_web_endpoint(request: FeaturesRequest):
    result = predict_web(request.features)
    return {**result, **calculate_security_score(result)}


@app.post("/predict/windows")
def predict_windows_endpoint(request: FeaturesRequest):
    result = predict_windows(request.features)
    return {**result, **calculate_security_score(result)}