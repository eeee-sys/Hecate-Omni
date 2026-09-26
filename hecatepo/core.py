"""Algorithm only; all rollout-derived quantities are detached from autograd.

Response positions use half-open intervals [start, end). A selected entropy
peak starts the next segment.
"""
from dataclasses import dataclass
import math
import random
import torch


@dataclass(frozen=True)
class AlgorithmConfig:
    ema_alpha: float = 0.3
    entropy_quantile: float = 0.60
    nms_distance: int = 64
    anchor_window: int = 8
    anchor_threshold: float = 0.90
    gamma: float = 1.0
    lambda_base: float = 0.5
    lambda_high: float = 0.5
    lambda_low: float = 0.5
    clip_epsilon: float = 0.2
    std_epsilon: float = 1e-6
    advantage_mode: str = "adaptive"
    anchor_mode: str = "semantic"

    def validate(self):
        if not 0 < self.ema_alpha <= 1 or not 0 <= self.entropy_quantile <= 1:
            raise ValueError("Invalid EMA or entropy quantile")
        if self.nms_distance < 1 or self.anchor_window < 1:
            raise ValueError("NMS distance and anchor window must be positive")
        if not -1 <= self.anchor_threshold <= 1:
            raise ValueError("anchor_threshold must be a cosine similarity")
        if self.gamma != 1.0:
            raise ValueError("v1 fixes gamma=1; discounted prompt fallback is not specified")
        if not 0 <= self.lambda_base <= 1 or min(self.lambda_high, self.lambda_low) < 0:
            raise ValueError("Invalid mixing parameters")
        if not 0 < self.clip_epsilon < 1 or self.std_epsilon <= 0:
            raise ValueError("Invalid numerical settings")
        if self.advantage_mode not in {"adaptive", "trajectory", "fixed"}:
            raise ValueError("Unknown advantage mode")
        if self.anchor_mode not in {"semantic", "random"}:
            raise ValueError("Unknown anchor mode")


@dataclass
class ResponseStats:
    log_probs: torch.Tensor
    entropy: torch.Tensor
    # Row t contains the state after consuming response token t (not predicting it).
    token_hidden: torch.Tensor


@dataclass
class Segment:
    trajectory: int
    start: int
    end: int
    entropy: float
    weight: float
    anchor: torch.Tensor | None
    root: bool
    group: int = -1
    semantic_group: int = -1
    group_size: int = 1
    trajectory_advantage: float = 0.0
    local_advantage: float = 0.0
    mixing: float = 0.0
    advantage: float = 0.0


def segment_spans(entropy: torch.Tensor, cfg: AlgorithmConfig):
    h = entropy.detach().float().cpu()
    if h.ndim != 1 or not h.numel() or not torch.isfinite(h).all():
        raise ValueError("Entropy must be a finite nonempty vector")
    smooth = h.clone()
    for t in range(1, len(h)):
        smooth[t] = cfg.ema_alpha * h[t] + (1 - cfg.ema_alpha) * smooth[t - 1]
    threshold = torch.quantile(smooth, cfg.entropy_quantile, interpolation="linear")
    peaks = [t for t in range(1, len(h) - 1)
             if smooth[t] >= threshold and smooth[t] >= smooth[t - 1]
             and smooth[t] > smooth[t + 1]]
    selected = []
    for t in sorted(peaks, key=lambda t: (-float(smooth[t]), t)):
        if all(abs(t - other) >= cfg.nms_distance for other in selected):
            selected.append(t)
    boundaries = [0, *sorted(selected), len(h)]
    return list(zip(boundaries[:-1], boundaries[1:]))


def _sample_normalize(values, eps):
    x = torch.tensor(values, dtype=torch.float64)
    if len(x) < 2 or not torch.isfinite(x).all():
        raise ValueError("Need at least two finite outcomes")
    return ((x - x.mean()) / (x.std(correction=1) + eps)).tolist()


def cluster_anchors(segments: list[Segment], threshold: float):
    """Deterministic constrained complete-link; no cross-prompt inputs permitted.

    Root group is separate. A non-root group has at most one segment per
    trajectory. Merge score is minimum pairwise cosine; ties use segment ids.
    """
    roots = [i for i, s in enumerate(segments) if s.root]
    clusters = [[i] for i, s in enumerate(segments) if not s.root]
    similarities = {}
    for a in range(len(segments)):
        if segments[a].anchor is None:
            continue
        for b in range(a + 1, len(segments)):
            if segments[b].anchor is not None:
                similarities[a, b] = float(torch.dot(segments[a].anchor, segments[b].anchor))
    while True:
        options = []
        for a in range(len(clusters)):
            ta = {segments[i].trajectory for i in clusters[a]}
            for b in range(a + 1, len(clusters)):
                if ta & {segments[i].trajectory for i in clusters[b]}:
                    continue
                score = min(similarities.get(tuple(sorted((i, j))), -math.inf)
                            for i in clusters[a] for j in clusters[b])
                if score >= threshold:
                    merged = tuple(sorted(clusters[a] + clusters[b]))
                    options.append((-score, merged, a, b))
        if not options:
            break
        _, merged, a, b = min(options)
        clusters = [c for index, c in enumerate(clusters) if index not in (a, b)]
        clusters.append(list(merged))
        clusters.sort(key=lambda c: tuple(c))
    return ([roots] if roots else []) + sorted(clusters, key=lambda c: tuple(c))


def randomize_anchor_groups(segments: list[Segment], clusters: list[list[int]], seed: int):
    """Shuffle non-root membership while keeping group sizes and trajectory constraints.

    The root group is unchanged. Constrained swaps preserve the original number
    and sizes of groups, isolating which segments are grouped from group-size
    effects. This local PRNG does not perturb rollout sampling.
    """
    rng = random.Random(seed)
    groups = [list(group) for group in clusters]
    first = 1 if groups and all(segments[i].root for i in groups[0]) else 0
    nonroot = [(g, p) for g in range(first, len(groups)) for p in range(len(groups[g]))]
    if len(nonroot) < 2:
        return groups
    for _ in range(100 * len(nonroot)):
        (a, ap), (b, bp) = rng.sample(nonroot, 2)
        if a == b:
            continue
        left, right = groups[a][ap], groups[b][bp]
        lt, rt = segments[left].trajectory, segments[right].trajectory
        if lt == rt:
            continue
        if any(segments[i].trajectory == rt for p, i in enumerate(groups[a]) if p != ap):
            continue
        if any(segments[i].trajectory == lt for p, i in enumerate(groups[b]) if p != bp):
            continue
        groups[a][ap], groups[b][bp] = right, left
    return groups


@torch.no_grad()
def build_segments(stats: list[ResponseStats], rewards: list[float], cfg: AlgorithmConfig,
                   random_seed: int | None = None):
    cfg.validate()
    if len(stats) != len(rewards) or len(stats) < 2:
        raise ValueError("A prompt group needs G>=2 responses and one reward each")
    ae = _sample_normalize(rewards, cfg.std_epsilon)
    segments = []
    for i, state in enumerate(stats):
        if len(state.log_probs) != len(state.entropy) or len(state.token_hidden) != len(state.entropy):
            raise ValueError("Old policy token arrays must align")
        spans = segment_spans(state.entropy, cfg)
        means = torch.tensor([state.entropy[a:b].float().mean() for a, b in spans])
        weights = torch.softmax(means, dim=0)
        mean_h = float(means.mean())
        for k, (start, end) in enumerate(spans):
            anchor = None
            if start:
                prior = state.token_hidden[max(0, start - cfg.anchor_window):start].float().mean(0).cpu()
                norm = prior.norm()
                if torch.isfinite(prior).all() and norm > cfg.std_epsilon:
                    anchor = prior / norm
            criticality = float(means[k]) / mean_h if mean_h > cfg.std_epsilon else 1.0
            if cfg.advantage_mode == "trajectory":
                lam = 0.0
            elif cfg.advantage_mode == "fixed":
                lam = cfg.lambda_base
            elif criticality >= 1:
                lam = min(cfg.lambda_base * (1 + cfg.lambda_high * (criticality - 1)), 1.0)
            else:
                lam = max(cfg.lambda_base * (1 - cfg.lambda_low * (1 - criticality)), 0.0)
            segments.append(Segment(i, start, end, float(means[k]), float(weights[k]), anchor,
                                    root=(start == 0), trajectory_advantage=ae[i], mixing=lam))
    clusters = cluster_anchors(segments, cfg.anchor_threshold)
    for group_id, members in enumerate(clusters):
        for index in members:
            segments[index].semantic_group = group_id
    if cfg.anchor_mode == "random":
        if random_seed is None:
            raise ValueError("Random anchor mode requires an explicit reproducible seed")
        clusters = randomize_anchor_groups(segments, clusters, random_seed)
    for group_id, members in enumerate(clusters):
        outcomes = [rewards[segments[j].trajectory] for j in members]
        local = _sample_normalize(outcomes, cfg.std_epsilon) if len(members) > 1 else None
        for k, index in enumerate(members):
            s = segments[index]
            s.group, s.group_size = group_id, len(members)
            s.local_advantage = local[k] if local is not None else s.trajectory_advantage
            s.advantage = s.mixing * s.local_advantage + (1 - s.mixing) * s.trajectory_advantage
    return [[s for s in segments if s.trajectory == i] for i in range(len(stats))]


def segment_loss(new_log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                 segments: list[Segment], clip_epsilon: float,
                 ratio_mode: str = "segment"):
    if new_log_probs.shape != old_log_probs.shape:
        raise ValueError("Old/new log probabilities must align")
    if ratio_mode not in {"segment", "token"}:
        raise ValueError("Unknown policy ratio mode")
    losses, ratios, clipped = [], [], []
    for s in segments:
        log_ratio = (new_log_probs[s.start:s.end].float()
                     - old_log_probs[s.start:s.end].to(new_log_probs.device).float())
        ratio = log_ratio.mean().exp() if ratio_mode == "segment" else log_ratio.exp()
        if not torch.isfinite(ratio).all():
            raise FloatingPointError("Nonfinite policy probability ratio")
        advantage = new_log_probs.new_tensor(s.advantage, dtype=torch.float32)
        surrogate = torch.minimum(ratio * advantage,
                                  ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage)
        # Both modes give one weighted contribution per segment. The token
        # ablation clips each token separately, then averages within segment.
        losses.append(-s.weight * surrogate.mean())
        ratios.append(ratio.detach().mean())
        clipped.append((torch.abs(ratio.detach() - 1) > clip_epsilon).float().mean())
    if not losses:
        raise ValueError("No segments")
    return torch.stack(losses).sum(), {
        "ratio_mean": float(torch.stack(ratios).mean()),
        "clip_fraction": float(torch.stack(clipped).mean()),
    }


def group_diagnostics(group: list[list[Segment]]):
    all_s = [s for response in group for s in response]
    nonroot = [s for s in all_s if not s.root]
    return {
        "segments_per_response": len(all_s) / len(group),
        "single_segment_fraction": sum(len(s) == 1 for s in group) / len(group),
        "nonroot_anchor_coverage": sum(s.group_size > 1 for s in nonroot) / max(len(nonroot), 1),
        "anchor_reassignment_fraction": sum(s.group != s.semantic_group for s in nonroot) / max(len(nonroot), 1),
        "local_advantage_difference": sum(abs(s.local_advantage - s.trajectory_advantage) for s in all_s) / len(all_s),
        "mixing_mean": sum(s.mixing for s in all_s) / len(all_s),
    }
