def calculate_security_score(result: dict) -> dict:
    score = 100

    if result.get("source_type") == "url" and result.get("is_attack"):
        score -= 20
    if result.get("source_type") == "network" and result.get("is_attack"):
        score -= 30
    if result.get("source_type") == "web" and result.get("is_attack"):
        score -= 25
    if result.get("source_type") == "windows" and result.get("is_attack"):
        score -= 25

    high_risk = {
        "DDoS", "DoS Hulk", "PortScan", "FTP-Patator", "SSH-Patator",
        "phishing_or_malicious_url", "web_attack", "windows_suspicious_activity"
    }
    if result.get("attack_type") in high_risk:
        score -= 15

    score = max(score, 0)
    if score >= 90:
        level = "Excellent"
    elif score >= 75:
        level = "Good"
    elif score >= 50:
        level = "Moderate Risk"
    else:
        level = "High Risk"

    return {"security_score": score, "security_level": level}
