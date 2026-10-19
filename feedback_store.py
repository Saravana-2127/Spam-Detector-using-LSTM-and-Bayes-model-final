"""feedback_store.py — logs your corrections; also remembers them per message."""
import csv, os
from datetime import datetime, timezone
from text_model import message_key

FEEDBACK_FILE = "feedback.csv"
FIELDNAMES = ["timestamp", "message", "predicted_label", "corrected_label"]


def _key(message):
    return message_key(message)


def log_feedback(message, predicted_label, corrected_label, path=FEEDBACK_FILE):
    exists = os.path.exists(path)
    try:
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if not exists:
                w.writeheader()
            w.writerow({"timestamp": datetime.now(timezone.utc).isoformat(), "message": message,
                        "predicted_label": predicted_label, "corrected_label": corrected_label})
    except PermissionError:
        print(f"[Warning] Could not access {path}. Close it if it is open in Excel.")


def load_feedback(path=FEEDBACK_FILE):
    if not os.path.exists(path):
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return [(r.get("message", ""), r.get("corrected_label", "")) for r in csv.DictReader(f)]
    except Exception:
        return []


def feedback_count(path=FEEDBACK_FILE):
    return len(load_feedback(path))


def corrected_label_for(message, path=FEEDBACK_FILE):
    """If you already corrected this exact message (ignoring links/numbers), return that label."""
    k, found = _key(message), None
    for m, lab in load_feedback(path):
        if lab in ("spam", "ham") and _key(m) == k:
            found = lab            # latest correction wins
    return found