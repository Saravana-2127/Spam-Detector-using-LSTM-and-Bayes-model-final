"""
LSTM spam classifier v2 (PyTorch).
v1 problems fixed: (a) final hidden state was taken AFTER the padding, so short messages
were mostly "remembered" as pad; now bidirectional + masked mean/max pooling ignores pads.
(b) no class weighting (spam is ~13%); now pos_weight. (c) max_len 30 truncated longer
messages; now 60. (d) shared preprocess() with all other models.
"""
import re, random
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from text_model import preprocess


def tokenize(text):
    return re.findall(r"[a-z']+", preprocess(text))


class Vocabulary:
    PAD, UNK = "<pad>", "<unk>"

    def __init__(self, messages, min_freq=2):
        c = Counter()
        for m in messages:
            c.update(tokenize(m))
        self.itos = [self.PAD, self.UNK] + [w for w, n in c.items() if n >= min_freq]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text, max_len=60):
        ids = [self.stoi.get(t, 1) for t in tokenize(text)[:max_len]]
        return ids + [0] * (max_len - len(ids))

    def __len__(self):
        return len(self.itos)


class _DS(Dataset):
    def __init__(self, X, y, vocab, max_len):
        self.x = [vocab.encode(m, max_len) for m in X]
        self.y = [float(v) for v in y]
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(self.x[i], dtype=torch.long), torch.tensor(self.y[i])


class LSTMSpamClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 4, 1)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)                       # (B,T,1)
        out, _ = self.lstm(self.embedding(x))               # (B,T,2H)
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        mx = out.masked_fill(~mask, -1.0).max(1).values     # LSTM outputs live in (-1,1)
        return self.fc(self.drop(torch.cat([mean, mx], -1))).squeeze(1)


def fit_lstm(X, y, epochs=10, batch_size=32, lr=2e-3, max_len=60, seed=0, verbose=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    vocab = Vocabulary(X)
    model = LSTMSpamClassifier(len(vocab))
    loader = DataLoader(_DS(X, y, vocab, max_len), batch_size=batch_size, shuffle=True)
    pos = float(np.sum(y)); neg = len(y) - pos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1)))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(1, epochs + 1):
        model.train(); tot = 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        if verbose:
            print(f"  LSTM epoch {ep}/{epochs} loss {tot/len(loader.dataset):.4f}")
    return model.eval(), vocab, max_len


def save_lstm(model, vocab, max_len, path="lstm_model.pt"):
    torch.save({"model_state": model.state_dict(), "vocab_itos": vocab.itos, "max_len": max_len, "arch": "v2"}, path)


def load_lstm(path="lstm_model.pt"):
    ck = torch.load(path, map_location="cpu")
    if ck.get("arch") != "v2":
        raise RuntimeError("lstm_model.pt is from the old architecture. Run: python train_all.py")
    vocab = Vocabulary.__new__(Vocabulary)
    vocab.itos = ck["vocab_itos"]; vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
    model = LSTMSpamClassifier(len(vocab.itos)); model.load_state_dict(ck["model_state"]); model.eval()
    return model, vocab, ck["max_len"]


def predict_lstm_prob(bundle, text):
    model, vocab, max_len = bundle
    x = torch.tensor([vocab.encode(text, max_len)], dtype=torch.long)
    with torch.no_grad():
        return torch.sigmoid(model(x)).item()