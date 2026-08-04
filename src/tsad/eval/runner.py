"""Shared evaluation loop for DDPM and reconstruction baselines."""

from pathlib import Path

import numpy as np
import torch

from tsad.data.tensor_utils import window_to_tensor
from tsad.eval.metrics import evaluate_windows
from tsad.eval.range_metrics import range_f1, vus_pr
from tsad.eval.threshold_select import select_threshold
from tsad.models import DDPM, LSTMAutoencoder, TranAD, TransformerAutoencoder, UNet1D
from tsad.scoring.channel import score_window_channels
from tsad.scoring.diffusion import score_window_dre, score_window_reconstruction, score_window_tranad


def load_model(name: str, ckpt_path: Path, cfg: dict, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    channels = ckpt.get("n_channels") or cfg["model"]["channels"]
    if name == "ddpm":
        model = UNet1D(channels=channels, base=cfg["model"]["base_channels"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        ddpm = DDPM(timesteps=ckpt.get("timesteps", cfg["model"]["timesteps"]))
        return model, ddpm, channels, ckpt
    if name == "transformer_ae":
        tcfg = ckpt.get("transformer_cfg") or cfg.get("model", {}).get("transformer_ae", {})
        model = TransformerAutoencoder(input_size=channels, **tcfg).to(device)
    elif name == "tranad":
        tcfg = dict(ckpt.get("tranad_cfg") or cfg.get("model", {}).get("tranad", {}))
        n_window = tcfg.pop("n_window", cfg["data"]["window_size"])
        model = TranAD(n_feats=channels, n_window=n_window, **tcfg).to(device)
    else:
        model = LSTMAutoencoder(input_size=channels).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, None, channels, ckpt


def score_windows(
    name: str,
    model,
    ddpm,
    windows: np.ndarray,
    device: torch.device,
    diffusion_steps: list[int],
    scoring_cfg: dict | None = None,
    show_progress: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray | None]]:
    from tqdm import tqdm

    point_scores_list, series_scores, channel_maps = [], [], []
    it = range(len(windows))
    if show_progress:
        it = tqdm(it, desc="Scoring", leave=False)

    scoring_cfg = scoring_cfg or {}
    recon_method = scoring_cfg.get("reconstruct_method", "ddpm")
    ddim_steps = scoring_cfg.get("ddim_steps")
    ddim_eta = scoring_cfg.get("ddim_eta", 0.0)

    for i in it:
        x = window_to_tensor(windows[i], device)
        if name == "ddpm":
            r = score_window_dre(
                model,
                ddpm,
                x,
                diffusion_steps,
                reconstruct_method=recon_method,
                ddim_steps=ddim_steps,
                ddim_eta=ddim_eta,
            )
            ch = score_window_channels(
                model,
                ddpm,
                x,
                diffusion_steps[len(diffusion_steps) // 2],
                reconstruct_method=recon_method,
                ddim_steps=ddim_steps,
                ddim_eta=ddim_eta,
            )
        elif name == "tranad":
            r = score_window_tranad(model, x)
            ch = None
        else:
            r = score_window_reconstruction(model, x)
            ch = None
        point_scores_list.append(r["point_scores"])
        series_scores.append(r["series_score"])
        channel_maps.append(ch)
    return point_scores_list, np.array(series_scores), channel_maps


def _sample_windows(windows: np.ndarray, max_n: int, seed: int = 42) -> np.ndarray:
    if max_n is None or len(windows) <= max_n:
        return windows
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(windows), size=max_n, replace=False)
    return windows[idx]


def _fit_threshold(
    name: str,
    model,
    ddpm,
    split,
    cfg: dict,
    device: torch.device,
    val_ps: list[np.ndarray],
) -> tuple[float, str]:
    ev = cfg.get("eval", {})
    method = ev.get("threshold_method", "f1")

    if method == "f1":
        val_scores = np.concatenate(val_ps)
        val_labels = np.concatenate(split.val_point_labels)
        return select_threshold("f1", val_scores, val_labels)

    # POT / percentile: fit on normal train reconstruction scores
    max_train = ev.get("pot_train_max_windows", 150)
    train_w = _sample_windows(split.train, max_train, seed=cfg.get("seed", 42))
    train_ps, _, _ = score_windows(
        name, model, ddpm, train_w, device, cfg["scoring"]["diffusion_steps"], cfg.get("scoring")
    )
    train_scores = np.concatenate(train_ps)
    return select_threshold(
        method,
        train_scores,
        pot_init_q=ev.get("pot_init_q", 0.98),
        percentile_q=ev.get("percentile_q", 95.0),
    )


def _segment_metrics(point_scores_list: list[np.ndarray], point_labels_list: list[np.ndarray], threshold: float) -> dict:
    all_scores = np.concatenate(point_scores_list)
    all_labels = np.concatenate(point_labels_list)
    pred = (all_scores >= threshold).astype(int)
    rm = range_f1(pred, all_labels)
    vm = vus_pr(all_scores, all_labels, threshold, max_bias=None)
    return {**rm, **vm}


def evaluate_model(
    name: str,
    ckpt_path: Path,
    split,
    cfg: dict,
    device: torch.device,
    threshold: float | None = None,
    show_progress: bool = False,
) -> dict:
    model, ddpm, _, _ = load_model(name, ckpt_path, cfg, device)
    steps = cfg["scoring"]["diffusion_steps"]
    scoring_cfg = cfg.get("scoring", {})

    val_ps, val_ss, _ = score_windows(
        name, model, ddpm, split.val, device, steps, scoring_cfg, show_progress
    )
    if threshold is None:
        threshold, threshold_method = _fit_threshold(name, model, ddpm, split, cfg, device, val_ps)
    else:
        threshold_method = "fixed"

    val_m = evaluate_windows(
        val_ps, list(split.val_point_labels), split.val_window_labels, val_ss, threshold=threshold
    )
    val_m["segment"] = _segment_metrics(val_ps, list(split.val_point_labels), threshold)

    te_ps, te_ss, _ = score_windows(
        name, model, ddpm, split.test, device, steps, scoring_cfg, show_progress
    )
    test_m = evaluate_windows(
        te_ps, list(split.test_point_labels), split.test_window_labels, te_ss, threshold=threshold
    )
    test_m["segment"] = _segment_metrics(te_ps, list(split.test_point_labels), threshold)

    return {
        "validation": val_m,
        "test": test_m,
        "threshold": threshold,
        "threshold_method": threshold_method,
    }
