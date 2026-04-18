from preprocessing.text_preprocess import preprocess_text_input
from inference.model_loader import windows_model


def rule_based_windows_detection(data: dict):
    text = str(data).lower()
    event_id = str(data.get("EventID", ""))

    # successful logon طبيعي
    if event_id == "4624":
        if any(x in text for x in ["services.exe", "svchost.exe", "nt authority", "system"]):
            return {
                "source_type": "windows",
                "prediction": "normal",
                "is_attack": False,
                "attack_type": None,
                "confidence": 0.95
            }
    
    if "powershell" in text and (
        "invoke-webrequest" in text or
        "downloadstring" in text or
        "iex" in text
    ):
        return {
            "source_type": "windows",
            "prediction": "attack",
            "is_attack": True,
            "attack_type": "powershell_attack",
            "confidence": 0.95
        }

    
    if "mimikatz" in text or "sekurlsa" in text:
        return {
            "source_type": "windows",
            "prediction": "attack",
            "is_attack": True,
            "attack_type": "credential_dumping",
            "confidence": 0.97
        }

    
    if "cmd.exe" in text and ("whoami" in text or "net user" in text):
        return {
            "source_type": "windows",
            "prediction": "attack",
            "is_attack": True,
            "attack_type": "reconnaissance_command",
            "confidence": 0.9
        }

    
    if "nc.exe" in text or "reverse shell" in text:
        return {
            "source_type": "windows",
            "prediction": "attack",
            "is_attack": True,
            "attack_type": "reverse_shell",
            "confidence": 0.92
        }

    return None


def predict_windows(data: dict) -> dict:
    # أولاً rules
    rule_result = rule_based_windows_detection(data)
    if rule_result is not None:
        return rule_result

    
    X = preprocess_text_input(data)
    pred = int(windows_model.predict(X)[0])

    confidence = None
    if hasattr(windows_model, "predict_proba"):
        try:
            confidence = float(windows_model.predict_proba(X)[0][1])
        except Exception:
            confidence = None

    return {
        "source_type": "windows",
        "prediction": "attack" if pred == 1 else "normal",
        "is_attack": bool(pred == 1),
        "attack_type": "windows_suspicious_activity" if pred == 1 else None,
        "confidence": round(confidence, 4) if confidence else None
    }