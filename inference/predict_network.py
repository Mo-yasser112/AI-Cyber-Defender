from preprocessing.network_preprocess import preprocess_network_input
from inference.model_loader import (
    network_anomaly_model, network_features,
    network_attack_model, network_classifier_features
)

def rule_based_network_detection(data: dict):
    flow_duration = float(data.get("Flow Duration", 0))
    total_fwd = float(data.get("Total Fwd Packets", 0))
    total_bwd = float(data.get("Total Backward Packets", 0))
    flow_bytes_s = float(data.get("Flow Bytes/s", 0))
    flow_packets_s = float(data.get("Flow Packets/s", 0))
    syn_count = float(data.get("SYN Flag Count", 0))
    ack_count = float(data.get("ACK Flag Count", 0))
    dst_port = float(data.get("Destination Port", 0))

    
    if (
        total_fwd > 1000 and
        total_bwd <= 10 and
        flow_packets_s > 10000 and
        syn_count > 50 and
        ack_count <= 5
    ):
        return {
            "source_type": "network",
            "prediction": "attack",
            "is_attack": True,
            "attack_type": "SYN_Flood_or_DDoS",
            "confidence": 0.95
        }

  
    if (
        total_fwd > 100 and
        total_bwd == 0 and
        syn_count > 20 and
        ack_count == 0 and
        flow_duration < 200
    ):
        return {
            "source_type": "network",
            "prediction": "attack",
            "is_attack": True,
            "attack_type": "PortScan",
            "confidence": 0.9
        }

    
    if (
        dst_port == 21 and
        total_fwd > 50 and
        flow_packets_s > 1000 and
        syn_count > 10
    ):
        return {
            "source_type": "network",
            "prediction": "attack",
            "is_attack": True,
            "attack_type": "FTP_BruteForce",
            "confidence": 0.88
        }

    
    if (
        dst_port == 22 and
        total_fwd > 50 and
        flow_packets_s > 1000 and
        syn_count > 10
    ):
        return {
            "source_type": "network",
            "prediction": "attack",
            "is_attack": True,
            "attack_type": "SSH_BruteForce",
            "confidence": 0.88
        }

    return None


def predict_network(data: dict) -> dict:
    # أولاً rules
    rule_result = rule_based_network_detection(data)
    if rule_result is not None:
        return rule_result

    X_anomaly = preprocess_network_input(data, network_features)

    if X_anomaly.sum(axis=1).iloc[0] == 0:
        return {
            "source_type": "network",
            "prediction": "suspicious",
            "is_attack": True,
            "attack_type": "invalid_or_missing_features",
            "confidence": 0.3
        }

    anomaly_pred = int(network_anomaly_model.predict(X_anomaly)[0])

    attack_type = None
    confidence = None

    if hasattr(network_anomaly_model, "predict_proba"):
        try:
            confidence = float(network_anomaly_model.predict_proba(X_anomaly)[0][1])
        except Exception:
            confidence = None

    if anomaly_pred == 1:
        X_cls = preprocess_network_input(data, network_classifier_features)
        attack_type = str(network_attack_model.predict(X_cls)[0])

    return {
        "source_type": "network",
        "prediction": "attack" if anomaly_pred == 1 else "benign",
        "is_attack": bool(anomaly_pred == 1),
        "attack_type": attack_type,
        "confidence": round(confidence, 4) if confidence is not None else None
    }