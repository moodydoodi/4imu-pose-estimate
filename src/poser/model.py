"""Network and loss.

Temporal convolution stem, bidirectional LSTM, two output stages (first the four
limb ends, then all twelve bones), forward kinematics turning directions into
positions.

Design notes:
  * The output is twelve unit vectors rather than 6D rotations or free
    positions. With fixed bone lengths that parameterisation is exact and bone
    lengths cannot become invalid.
  * Two stages. The four bones carrying a sensor are the most directly observed;
    they get their own intermediate output and loss term, and the second stage
    may use them (leaf-to-full, as in TransPose).
  * BiLSTM rather than a transformer or ST-GCN. With about one hour of real data
    model capacity is not the bottleneck. Bidirectional means offline only,
    which suits the evaluation setting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LEAF_BONES, N_BONES, N_FEAT, N_JOINTS, PARENTS


def fk(dirs, lengths):
    """dirs (B,T,12,3), lengths (12,) or (B,12) -> (B,T,13,3)."""
    B, T = dirs.shape[:2]
    d = F.normalize(dirs, dim=-1, eps=1e-8)
    if lengths.dim() == 1:
        L = lengths.view(1, 1, N_BONES, 1).expand(B, T, N_BONES, 1)
    else:
        L = lengths.view(B, 1, N_BONES, 1).expand(B, T, N_BONES, 1)
    seg = d * L
    P = [torch.zeros(B, T, 3, dtype=dirs.dtype, device=dirs.device)]
    for i in range(1, N_JOINTS):
        P.append(P[PARENTS[i]] + seg[:, :, i - 1])
    return torch.stack(P, dim=2)


class TemporalStem(nn.Module):
    """Dilations cover about 0.4 s, the duration of a landing."""

    def __init__(self, cin, cout, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, 5, padding=2), nn.GELU(), nn.BatchNorm1d(cout),
            nn.Conv1d(cout, cout, 5, padding=4, dilation=2), nn.GELU(), nn.BatchNorm1d(cout),
            nn.Conv1d(cout, cout, 5, padding=8, dilation=4), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class Poser(nn.Module):
    def __init__(self, bone_lengths, hidden=256, layers=2, dropout=0.2,
                 stem=128, n_feat=N_FEAT):
        super().__init__()
        self.register_buffer("L", torch.as_tensor(bone_lengths, dtype=torch.float32))
        self.inorm = nn.LayerNorm(n_feat)
        self.stem = TemporalStem(n_feat, stem, dropout)
        self.rnn1 = nn.LSTM(stem, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head_leaf = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                       nn.Linear(hidden, len(LEAF_BONES) * 3))
        self.rnn2 = nn.LSTM(2 * hidden + len(LEAF_BONES) * 3, hidden, 1,
                            batch_first=True, bidirectional=True)
        self.head_full = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(hidden, N_BONES * 3))

    def forward(self, x, lengths=None):
        B, T, _ = x.shape
        L = self.L if lengths is None else lengths
        h = self.stem(self.inorm(x))
        h, _ = self.rnn1(h)

        leaf = self.head_leaf(h).view(B, T, len(LEAF_BONES), 3)
        leaf_n = F.normalize(leaf, dim=-1, eps=1e-8)

        h2, _ = self.rnn2(torch.cat([h, leaf_n.reshape(B, T, -1)], dim=-1))
        dirs = F.normalize(self.head_full(h2).view(B, T, N_BONES, 3), dim=-1, eps=1e-8)

        # Same kinematics with only the four leaf bones replaced by stage one, so
        # its loss term is measurable in mm. The other bones are detached so this
        # term trains stage one alone. Built with stack rather than in-place
        # assignment to keep autograd happy.
        leaf_map = {b - 1: k for k, b in enumerate(LEAF_BONES)}
        dirs_leaf = torch.stack(
            [leaf_n[:, :, leaf_map[i]] if i in leaf_map else dirs[:, :, i].detach()
             for i in range(N_BONES)], dim=2)

        return {"pos": fk(dirs, L), "pos_leaf": fk(dirs_leaf, L),
                "dirs": dirs, "dirs_leaf": leaf_n}


class PoseLoss(nn.Module):
    """Position, direction, velocity and the intermediate stage."""

    def __init__(self, w_dir=0.5, w_vel=1.0, w_leaf=0.5, huber=0.05):
        super().__init__()
        self.w_dir, self.w_vel, self.w_leaf = w_dir, w_vel, w_leaf
        self.huber = huber

    def forward(self, out, y_pos, y_dirs):
        h = lambda a, b: F.huber_loss(a, b, delta=self.huber)
        l_pos = h(out["pos"], y_pos)
        l_dir = (1.0 - (out["dirs"] * y_dirs).sum(-1)).mean()
        l_vel = h(out["pos"][:, 1:] - out["pos"][:, :-1], y_pos[:, 1:] - y_pos[:, :-1])
        l_leaf = h(out["pos_leaf"], y_pos)
        total = l_pos + self.w_dir * l_dir + self.w_vel * l_vel + self.w_leaf * l_leaf
        return total, {"pos": l_pos.item(), "dir": l_dir.item(),
                       "vel": l_vel.item(), "leaf": l_leaf.item()}
