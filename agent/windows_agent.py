import time
import json
import requests
import win32evtlog

API_URL = "http://127.0.0.1:8000/predict/windows"
LOG_TYPE = "Security"
SERVER = "localhost"

POLL_INTERVAL = 5
MAX_EVENTS_PER_BATCH = 20


WATCHED_EVENT_IDS = {4624, 4625, 4688, 4720, 4728, 1102}

SAFE_EVENT_IDS = {4624}

last_record_number = None


def is_benign_known_event(event_data: dict) -> bool:
    event_id = event_data.get("EventID")
    inserts = str(event_data.get("StringInserts", "")).lower()

    # Successful logon طبيعي
    if event_id == 4624:
        if any(x in inserts for x in ["services.exe", "svchost.exe", "system", "nt authority"]):
            return True

    return False


def read_new_events():
    global last_record_number

    hand = win32evtlog.OpenEventLog(SERVER, LOG_TYPE)
    flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
    events = win32evtlog.ReadEventLog(hand, flags, 0)

    new_events = []

    if events:
        for event in events[:MAX_EVENTS_PER_BATCH]:
            record_number = event.RecordNumber

            if last_record_number is not None and record_number <= last_record_number:
                continue

            event_id = event.EventID & 0xFFFF

            if event_id not in WATCHED_EVENT_IDS:
                continue

            event_data = {
                "EventID": event_id,
                "SourceName": event.SourceName,
                "EventCategory": event.EventCategory,
                "TimeGenerated": str(event.TimeGenerated),
                "ComputerName": event.ComputerName,
                "RecordNumber": record_number,
                "EventType": event.EventType,
                "StringInserts": " | ".join(event.StringInserts) if event.StringInserts else ""
            }

            if is_benign_known_event(event_data):
                continue

            new_events.append(event_data)

        if new_events:
            last_record_number = max(e["RecordNumber"] for e in new_events)

    return new_events


def send_to_api(event_data):
    payload = {
        "features": event_data
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] Failed to send event: {e}")
        return None


def main():
    print(f"[INFO] Windows Agent started. Monitoring '{LOG_TYPE}' log...")
    print(f"[INFO] Sending results to: {API_URL}")

    while True:
        try:
            events = read_new_events()

            for event in events:
                result = send_to_api(event)

                if result:
                    print("\n[EVENT]")
                    print(json.dumps(event, indent=2, ensure_ascii=False))

                    print("[AI RESULT]")
                    print(json.dumps(result, indent=2, ensure_ascii=False))

                    # alert بس لو الهجوم واضح فعلًا
                    if result.get("is_attack") and float(result.get("confidence") or 0) >= 0.90:
                        print("[ALERT] Suspicious Windows activity detected!")

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("\n[INFO] Agent stopped.")
            break

        except Exception as e:
            print(f"[ERROR] Agent crashed: {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()