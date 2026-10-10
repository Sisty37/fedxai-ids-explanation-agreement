import torch
from torch import nn


class MLP(nn.Module):
    """d -> 128 -> 64 -> C, ReLU, dropout 0.2, no batch norm (locked)."""

    def __init__(self, d: int, n_classes: int, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(64, n_classes))

    def forward(self, x):
        return self.net(x)


def get_params(model):
    return [p.detach().clone() for p in model.parameters()]


def set_params(model, params):
    with torch.no_grad():
        for p, q in zip(model.parameters(), params):
            p.copy_(q)


class Conv1dMM(nn.Module):
    """1D convolution (stride 1, 'same' padding) computed as sliding windows + one
    linear layer, i.e. a matrix multiplication. Mathematically a standard Conv1d;
    avoids slow cuDNN convolution paths on some Windows/laptop GPU setups.
    Input/output layout: (B, L, C)."""

    def __init__(self, cin: int, cout: int, k: int):
        super().__init__()
        self.k, self.lin = k, nn.Linear(cin * k, cout)

    def forward(self, x):
        p = self.k // 2
        x = torch.nn.functional.pad(x, (0, 0, p, p))          # pad sequence dim
        w = x.unfold(1, self.k, 1)                             # (B, L, C, k)
        return self.lin(w.reshape(w.shape[0], w.shape[1], -1))


class CNN1D(nn.Module):
    """1D-CNN over the feature vector (second architecture for validation).
    (B, d) -> Conv(1->32, k5) -> ReLU -> Conv(32->32, k5) -> ReLU -> MaxPool(2)
    -> FC(32*d/2 -> 64) -> ReLU -> Dropout(0.2) -> FC(64 -> C)."""

    def __init__(self, d: int, n_classes: int, p: float = 0.2):
        super().__init__()
        self.c1, self.c2 = Conv1dMM(1, 32, 5), Conv1dMM(32, 32, 5)
        self.head = nn.Sequential(nn.Linear(32 * (d // 2), 64), nn.ReLU(), nn.Dropout(p), nn.Linear(64, n_classes))

    def forward(self, x):
        h = torch.relu(self.c2(torch.relu(self.c1(x.unsqueeze(-1)))))   # (B, d, 32)
        B, L, C = h.shape
        h = h[:, : (L // 2) * 2].reshape(B, L // 2, 2, C).amax(2)      # max-pool over pairs
        return self.head(h.reshape(B, -1))


def build_model(arch: str, d: int, n_classes: int):
    return {"mlp": MLP, "cnn": CNN1D}[arch](d, n_classes)
