"""Paired comparison of two variants with bootstrap 95 % intervals.
Pairs runs by (seed, LORO test recording) and refuses to compare runs that do
not pair up. Writes .csv, .json and .pdf. This produced the files in results/.

    python src/poser/compare_metrics_to_pdf.py \
        --baseline 'models/base_s*/metrics.json' \
        --experiment 'models/ft_s*/metrics.json' \
        --out results/ft_vs_real.pdf
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

METRICS = ("mpjpe", "pa_mpjpe", "pck100")
LABELS = {"mpjpe": "MPJPE (mm)", "pa_mpjpe": "PA-MPJPE (mm)", "pck100": "PCK@100 (%)"}


def expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        found = [Path(p) for p in glob.glob(pattern, recursive=True)]
        if found:
            paths.extend(found)
        elif Path(pattern).is_file():
            paths.append(Path(pattern))
        else:
            raise FileNotFoundError(f"No metrics.json matching: {pattern}")
    unique = sorted({p.resolve() for p in paths})
    if not unique:
        raise FileNotFoundError("No metrics.json found.")
    return unique


def load_runs(paths: list[Path], role: str) -> tuple[dict[tuple[int, str], dict], list[str]]:
    pairs: dict[tuple[int, str], dict] = {}
    warnings: list[str] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        seed = data.get("args", {}).get("seed")
        if seed is None:
            raise ValueError(f"{path}: args.seed is missing.")
        card_path = path.parent / "model_card.json"
        frame = None
        if card_path.exists():
            frame = json.loads(card_path.read_text(encoding="utf-8")).get("frame")
        else:
            warnings.append(f"{role}: model_card.json missing next to {path.parent}")
        folds = data.get("folds")
        if not isinstance(folds, list) or not folds:
            raise ValueError(f"{path}: folds is missing or empty.")
        for fold in folds:
            test = fold.get("test")
            if not test:
                raise ValueError(f"{path}: fold without a test name.")
            key = (int(seed), str(test))
            if key in pairs:
                raise ValueError(f"Duplicate pair {key} in {role}: {pairs[key]['source']} and {path}")
            missing = [m for m in METRICS if m not in fold]
            if missing:
                raise ValueError(f"{path}, {key}: metrics missing: {missing}")
            pairs[key] = {"seed": int(seed), "test": str(test), "frame": frame,
                          "source": str(path), **{m: float(fold[m]) for m in METRICS}}
    return pairs, warnings


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n: int = 20_000) -> tuple[float, float]:
    samples = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def paired_rows(base: dict, experiment: dict) -> list[dict]:
    kb, ke = set(base), set(experiment)
    if kb != ke:
        only_b = sorted(kb - ke)
        only_e = sorted(ke - kb)
        raise ValueError("Runs cannot be paired. Only in baseline: " + str(only_b[:8]) +
                         "; only in experiment: " + str(only_e[:8]))
    rows = []
    for key in sorted(kb):
        b, e = base[key], experiment[key]
        if b["frame"] and e["frame"] and b["frame"] != e["frame"]:
            raise ValueError(f"{key}: frame mismatch {b['frame']} vs. {e['frame']}")
        row = {"seed": b["seed"], "test": b["test"], "frame": b["frame"] or e["frame"] or "unknown"}
        for metric in METRICS:
            row[f"baseline_{metric}"] = b[metric]
            row[f"experiment_{metric}"] = e[metric]
            # Lower is better for errors, higher for PCK.
            row[f"delta_{metric}"] = e[metric] - b[metric]
            row[f"improves_{metric}"] = (e[metric] < b[metric]) if metric != "pck100" else (e[metric] > b[metric])
        rows.append(row)
    return rows


def summarise(rows: list[dict]) -> dict:
    rng = np.random.default_rng(20260808)
    summary = {"n_pairs": len(rows), "seeds": sorted({r["seed"] for r in rows}),
               "folds": sorted({r["test"] for r in rows}), "metrics": {}}
    for metric in METRICS:
        base = np.array([r[f"baseline_{metric}"] for r in rows])
        exp = np.array([r[f"experiment_{metric}"] for r in rows])
        delta = exp - base
        lo, hi = bootstrap_ci(delta, rng)
        summary["metrics"][metric] = {
            "baseline_mean": float(base.mean()), "experiment_mean": float(exp.mean()),
            "delta_mean": float(delta.mean()), "delta_median": float(np.median(delta)),
            "delta_std": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
            "bootstrap_95_ci": [lo, hi],
            "improved_pairs": int(sum(r[f"improves_{metric}"] for r in rows)),
            "total_pairs": len(rows),
        }
    return summary


def render_pdf(out: Path, summary: dict, rows: list[dict], base_paths: list[Path], exp_paths: list[Path], warnings: list[str]) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    except ImportError as exc:
        raise SystemExit("reportlab is missing. Install it with: python -m pip install reportlab") from exc

    out.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    doc = SimpleDocTemplate(str(out), pagesize=A4, rightMargin=15*mm, leftMargin=15*mm,
                            topMargin=14*mm, bottomMargin=14*mm)
    story = [Paragraph("Paired Synthetic Pretraining Comparison", styles["Title"]),
             Paragraph("Real-only baseline versus synthetic-pretrained and real-fine-tuned model.", styles["BodyText"]),
             Spacer(1, 4*mm),
             Paragraph(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}<br/>"
                       f"Paired observations: {summary['n_pairs']} = {len(summary['seeds'])} seeds x "
                       f"{len(summary['folds'])} LORO folds.<br/>"
                       "Delta = experiment - baseline. Negative error delta is better; positive PCK delta is better.",
                       styles["BodyText"]), Spacer(1, 5*mm)]

    head = ["Metric", "Baseline mean", "Experiment mean", "Mean delta", "95% bootstrap CI", "Improved"]
    data = [head]
    for metric in METRICS:
        x = summary["metrics"][metric]
        ci = x["bootstrap_95_ci"]
        data.append([LABELS[metric], f"{x['baseline_mean']:.2f}", f"{x['experiment_mean']:.2f}",
                     f"{x['delta_mean']:+.2f}", f"[{ci[0]:+.2f}, {ci[1]:+.2f}]",
                     f"{x['improved_pairs']}/{x['total_pairs']}"])
    table = Table(data, colWidths=[33*mm, 25*mm, 28*mm, 24*mm, 39*mm, 19*mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B8C4CE")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EDF3F7")]),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [Paragraph("Primary result", styles["Heading2"]), table, Spacer(1, 5*mm)]

    mp = summary["metrics"]["mpjpe"]
    verdict = "improvement" if mp["delta_mean"] < 0 else "no mean improvement"
    story.append(Paragraph(f"MPJPE verdict: <b>{verdict}</b>; mean delta {mp['delta_mean']:+.2f} mm, "
                           f"95% bootstrap interval [{mp['bootstrap_95_ci'][0]:+.2f}, {mp['bootstrap_95_ci'][1]:+.2f}] mm.",
                           styles["BodyText"]))
    story.append(Spacer(1, 5*mm))

    story.append(Paragraph("Paired fold results", styles["Heading2"]))
    detail = [["Seed", "Test", "Base MPJPE", "Fine-tuned", "Delta", "Base PA", "Fine-tuned", "Delta", "PCK delta"]]
    for r in rows:
        detail.append([str(r["seed"]), r["test"], f"{r['baseline_mpjpe']:.1f}", f"{r['experiment_mpjpe']:.1f}",
                       f"{r['delta_mpjpe']:+.1f}", f"{r['baseline_pa_mpjpe']:.1f}",
                       f"{r['experiment_pa_mpjpe']:.1f}", f"{r['delta_pa_mpjpe']:+.1f}", f"{r['delta_pck100']:+.1f}"])
    detail_table = Table(detail, colWidths=[12*mm, 25*mm, 20*mm, 20*mm, 15*mm, 17*mm, 20*mm, 15*mm, 17*mm], repeatRows=1)
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#B8C4CE")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EDF3F7")]),
        ("ALIGN", (2, 1), (-1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story += [detail_table, Spacer(1, 4*mm)]
    provenance = "<br/>".join(["Baseline inputs:"] + [str(p) for p in base_paths] +
                               ["Experiment inputs:"] + [str(p) for p in exp_paths])
    story.append(Paragraph(provenance, styles["Small"]))
    if warnings:
        story.append(Spacer(1, 3*mm)); story.append(Paragraph("Warnings: " + "; ".join(warnings), styles["Small"]))
    doc.build(story)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired metrics.json comparison -> PDF report")
    ap.add_argument("--baseline", nargs="+", required=True, help="Glob(s) or metrics.json files for real-only runs")
    ap.add_argument("--experiment", nargs="+", required=True, help="Glob(s) or metrics.json files for fine-tuned runs")
    ap.add_argument("--out", required=True, help="Output PDF path")
    args = ap.parse_args()
    base_paths, exp_paths = expand(args.baseline), expand(args.experiment)
    base, wb = load_runs(base_paths, "baseline")
    experiment, we = load_runs(exp_paths, "experiment")
    rows = paired_rows(base, experiment)
    summary = summarise(rows)
    out = Path(args.out)
    render_pdf(out, summary, rows, base_paths, exp_paths, wb + we)
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "pairs": rows}, indent=2), encoding="utf-8")
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"PDF: {out}\nJSON: {out.with_suffix('.json')}\nCSV: {out.with_suffix('.csv')}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
