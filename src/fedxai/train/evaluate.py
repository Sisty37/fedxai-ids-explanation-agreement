import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support


@torch.no_grad()
def predict(model, X, idx, batch=65536):
    model.eval()
    return torch.cat([model(X[idx[s:s + batch]]).argmax(1) for s in range(0, len(idx), batch)]).cpu().numpy()


def detection_metrics(model, X, y, idx, n_classes):
    yp = predict(model, X, idx)
    yt = y[idx].cpu().numpy()
    labels = list(range(n_classes))
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    return {"macro_f1": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
            "per_class_precision": p.tolist(), "per_class_recall": r.tolist(),
            "per_class_f1": f.tolist(), "support": s.tolist()}
