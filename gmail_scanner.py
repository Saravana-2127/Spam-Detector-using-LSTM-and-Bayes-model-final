"""
Gmail Spam/Fraud Scanner
=========================
Connects to your Gmail inbox (official Gmail API, OAuth — no password sharing),
pulls recent/unread emails, and runs each one through both trained models
(Naive Bayes + LSTM) to flag spam/fraud.

First run: a browser window will open asking you to log in and approve access.
After that, a `token.json` file is saved so you won't need to log in again.

Usage:
    python3 gmail_scanner.py            # scan the 10 most recent emails
    python3 gmail_scanner.py --unread   # scan only unread emails
    python3 gmail_scanner.py --label    # also apply a "Spam-Alert" Gmail label to flagged emails
"""

import argparse
import base64
import os
import pickle
from email import message_from_bytes

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

import torch
from naive_bayes import NaiveBayesSpamClassifier
from lstm_model import LSTMSpamClassifier, Vocabulary

# Gmail API scopes: read messages + modify labels
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]

LABEL_NAME = "Spam-Alert"


def get_gmail_service():
    """Handles OAuth login (first time) and returns an authenticated Gmail API client."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token_file:
            token_file.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def load_models():
    """Load the trained Naive Bayes and LSTM models."""
    nb_model = NaiveBayesSpamClassifier.load("naive_bayes_model.pkl")

    checkpoint = torch.load("lstm_model.pt", map_location="cpu")
    vocab = Vocabulary.__new__(Vocabulary)
    vocab.itos = checkpoint["vocab_itos"]
    vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
    max_len = checkpoint["max_len"]

    lstm_model = LSTMSpamClassifier(vocab_size=len(vocab.itos))
    lstm_model.load_state_dict(checkpoint["model_state"])
    lstm_model.eval()

    return nb_model, (lstm_model, vocab, max_len)


def predict_naive_bayes(nb_model, text):
    return nb_model.predict_proba(text).get("spam", 0.0)


def predict_lstm(lstm_bundle, text):
    lstm_model, vocab, max_len = lstm_bundle
    ids = vocab.encode(text, max_len)
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        return torch.sigmoid(lstm_model(x)).item()


def get_message_text(service, msg_id):
    """Fetch a message and extract subject + plain-text body snippet."""
    raw = service.users().messages().get(userId="me", id=msg_id, format="raw").execute()
    msg_bytes = base64.urlsafe_b64decode(raw["raw"])
    mime_msg = message_from_bytes(msg_bytes)

    subject = mime_msg.get("Subject", "(no subject)")
    sender = mime_msg.get("From", "(unknown sender)")

    body = ""
    if mime_msg.is_multipart():
        for part in mime_msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    body = part.get_payload(decode=True).decode(errors="ignore")
                except Exception:
                    pass
                break
    else:
        try:
            body = mime_msg.get_payload(decode=True).decode(errors="ignore")
        except Exception:
            body = raw.get("snippet", "")

    combined = f"{subject}\n{body}"
    return subject, sender, combined[:2000]  # cap length for speed


def get_or_create_label(service, label_name):
    labels = service.users().labels().list(userId="me").execute().get("labels", [])
    for label in labels:
        if label["name"] == label_name:
            return label["id"]
    new_label = service.users().labels().create(
        userId="me",
        body={"name": label_name, "labelListVisibility": "labelShow", "messageListVisibility": "show"},
    ).execute()
    return new_label["id"]


def scan_inbox(max_results=10, unread_only=False, apply_label=False):
    service = get_gmail_service()
    nb_model, lstm_bundle = load_models()

    query = "is:unread" if unread_only else ""
    results = service.users().messages().list(
        userId="me", labelIds=["INBOX"], q=query, maxResults=max_results
    ).execute()
    messages = results.get("messages", [])

    if not messages:
        print("No messages found.")
        return

    label_id = get_or_create_label(service, LABEL_NAME) if apply_label else None

    print(f"Scanning {len(messages)} email(s)...\n")
    for m in messages:
        subject, sender, text = get_message_text(service, m["id"])
        nb_prob = predict_naive_bayes(nb_model, text)
        lstm_prob = predict_lstm(lstm_bundle, text)
        avg_prob = (nb_prob + lstm_prob) / 2
        verdict = "SPAM/SUSPICIOUS" if avg_prob >= 0.5 else "looks safe"

        flag = "🚨" if avg_prob >= 0.5 else "✅"
        print(f"{flag} From: {sender}")
        print(f"   Subject: {subject}")
        print(f"   Naive Bayes: {nb_prob*100:.1f}%  |  LSTM: {lstm_prob*100:.1f}%  ->  {verdict}")
        print()

        if apply_label and avg_prob >= 0.5:
            service.users().messages().modify(
                userId="me", id=m["id"], body={"addLabelIds": [label_id]}
            ).execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan Gmail inbox for spam/fraud.")
    parser.add_argument("--unread", action="store_true", help="Only scan unread emails")
    parser.add_argument("--label", action="store_true", help="Apply 'Spam-Alert' Gmail label to flagged emails")
    parser.add_argument("--max", type=int, default=10, help="Max number of emails to scan")
    args = parser.parse_args()

    scan_inbox(max_results=args.max, unread_only=args.unread, apply_label=args.label)
