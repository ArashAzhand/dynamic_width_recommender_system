"""Report Synchronization & Compilation Engine.

Reads benchmark_results.csv, updates LaTeX tables and figures, and compiles main.pdf.
"""

import os
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_mean_std(val_str):
    """Extracts mean and std from string 'mean +/- std'."""
    if isinstance(val_str, str) and "+/-" in val_str:
        parts = val_str.split("+/-")
        return float(parts[0].strip()), float(parts[1].strip())
    try:
        val = float(val_str)
        return val, 0.0
    except Exception:
        return 0.0, 0.0


def generate_latex_table(csv_path: str, output_tex: str):
    """Generates a professional publication-grade LaTeX table from benchmark CSV."""
    df = pd.read_csv(csv_path)

    table_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Empirical Recommendation Benchmark Results on MovieLens-100K across 4 GNN Backbones and 6 Configurations under Early Stopping (Patience = 15 eval checks / 75 epochs, Ceiling = 300 epochs). Highest scores within each backbone are highlighted in \textbf{bold}.}",
        r"\label{tab:benchmark_results}",
        r"\small",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"\textbf{Backbone} & \textbf{Configuration} & \textbf{Recall@10} & \textbf{Recall@20} & \textbf{NDCG@10} & \textbf{NDCG@20} & \textbf{MAP@20} & \textbf{MRR@20} & \textbf{Best Epoch} \\",
        r"\midrule",
    ]

    current_backbone = None

    for _, row in df.iterrows():
        cfg_full = row["Model Configuration"]
        backbone, variant = cfg_full.split(" - ")

        if current_backbone is not None and backbone != current_backbone:
            table_lines.append(r"\midrule")
        current_backbone = backbone

        # Format variants
        var_display = variant.replace("_", r"\_")
        if "PANDA-" in var_display:
            var_display = r"\textbf{" + var_display + r"}"

        r10_m, _ = extract_mean_std(row["Recall@10"])
        r20_m, _ = extract_mean_std(row["Recall@20"])
        n10_m, _ = extract_mean_std(row["NDCG@10"])
        n20_m, _ = extract_mean_std(row["NDCG@20"])
        map20_m, _ = extract_mean_std(row["MAP@20"])
        mrr20_m, _ = extract_mean_std(row["MRR@20"])
        ep_m, _ = extract_mean_std(row["Best_Epoch"])

        # Check if max in backbone
        b_df = df[df["Model Configuration"].str.startswith(backbone + " - ")]
        max_r20 = max([extract_mean_std(v)[0] for v in b_df["Recall@20"]])
        max_n20 = max([extract_mean_std(v)[0] for v in b_df["NDCG@20"]])

        r10_str = f"{r10_m:.4f}"
        r20_str = f"\\textbf{{{r20_m:.4f}}}" if abs(r20_m - max_r20) < 1e-5 else f"{r20_m:.4f}"
        n10_str = f"{n10_m:.4f}"
        n20_str = f"\\textbf{{{n20_m:.4f}}}" if abs(n20_m - max_n20) < 1e-5 else f"{n20_m:.4f}"
        map_str = f"{map20_m:.4f}"
        mrr_str = f"{mrr20_m:.4f}"
        ep_str = f"{ep_m:.1f}"

        line = f"{backbone} & {var_display} & {r10_str} & {r20_str} & {n10_str} & {n20_str} & {map_str} & {mrr_str} & {ep_str} \\\\"
        table_lines.append(line)

    table_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ])

    os.makedirs(os.path.dirname(output_tex), exist_ok=True)
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines) + "\n")
    print(f"Generated LaTeX results table: {output_tex}")


def update_figures(csv_path: str, fig_dir: str):
    """Regenerates figures based on current benchmark CSV."""
    os.makedirs(fig_dir, exist_ok=True)
    if os.path.exists("plots/user_degree_cutoff.png"):
        shutil.copy("plots/user_degree_cutoff.png", os.path.join(fig_dir, "user_degree_cutoff.png"))
    if os.path.exists("plots/item_degree_cutoff.png"):
        shutil.copy("plots/item_degree_cutoff.png", os.path.join(fig_dir, "item_degree_cutoff.png"))

    df = pd.read_csv(csv_path)
    df["Rec20"] = df["Recall@20"].apply(lambda v: extract_mean_std(v)[0])
    df["NDCG20"] = df["NDCG@20"].apply(lambda v: extract_mean_std(v)[0])
    df["Epoch"] = df["Best_Epoch"].apply(lambda v: extract_mean_std(v)[0])

    df["Backbone"] = df["Model Configuration"].apply(lambda x: x.split(" - ")[0])
    df["Variant"] = df["Model Configuration"].apply(lambda x: x.split(" - ")[1])

    backbones = ["LIGHTGCN", "GCN", "GRAPHSAGE", "GAT"]
    variants = [
        "Vanilla (d=64)", "Param-Matched (d=80)", "PANDA-Random",
        "PANDA-Degree", "PANDA-PageRank", "PANDA-Betweenness"
    ]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    x = np.arange(len(backbones))
    width = 0.13
    colors = ["#7f7f7f", "#1f77b4", "#e377c2", "#2ca02c", "#ff7f0e", "#d62728"]

    for i, var in enumerate(variants):
        rec_vals = [df[(df["Backbone"] == b) & (df["Variant"] == var)]["Rec20"].values[0] for b in backbones]
        ndcg_vals = [df[(df["Backbone"] == b) & (df["Variant"] == var)]["NDCG20"].values[0] for b in backbones]

        ax1.bar(x + i * width - 2.5 * width, rec_vals, width, label=var, color=colors[i], edgecolor="black", linewidth=0.5)
        ax2.bar(x + i * width - 2.5 * width, ndcg_vals, width, label=var, color=colors[i], edgecolor="black", linewidth=0.5)

    ax1.set_title("Test Recall@20 Across GNN Backbones", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Recall@20", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(backbones, fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 0.30)
    ax1.legend(loc="upper right", frameon=True, fontsize=8)

    ax2.set_title("Test NDCG@20 Across GNN Backbones", fontsize=12, fontweight="bold")
    ax2.set_ylabel("NDCG@20", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(backbones, fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 0.25)
    ax2.legend(loc="upper right", frameon=True, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "backbone_performance_comparison.png"))
    plt.close()

    # Convergence plot
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    for i, var in enumerate(variants):
        epoch_vals = [df[(df["Backbone"] == b) & (df["Variant"] == var)]["Epoch"].values[0] for b in backbones]
        ax.bar(x + i * width - 2.5 * width, epoch_vals, width, label=var, color=colors[i], edgecolor="black", linewidth=0.5)

    ax.set_title("Optimal Convergence Epochs Under Early Stopping (Patience = 15 checks / 75 epochs)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Best Epoch (Validation Recall@20)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(backbones, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 320)
    ax.axhline(140, color="gray", linestyle="--", alpha=0.6, label="Vanilla LightGCN (140 ep)")
    ax.legend(loc="upper right", frameon=True, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "convergence_epochs.png"))
    plt.close()
    print(f"Updated figures in {fig_dir}")


def compile_latex(report_dir: str = "Report", main_tex: str = "main.tex"):
    """Compiles the LaTeX document to PDF using pdflatex."""
    pdflatex_path = r"C:\Users\arash\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
    if not os.path.exists(pdflatex_path):
        pdflatex_path = "pdflatex"

    cmd = [pdflatex_path, "-interaction=nonstopmode", "-disable-installer", main_tex]
    print(f"Compiling LaTeX document: {main_tex} in {report_dir}...")

    # Pass 1
    res1 = subprocess.run(cmd, cwd=report_dir, capture_output=True, text=True)
    # Pass 2 for cross-references and figure numbers
    res2 = subprocess.run(cmd, cwd=report_dir, capture_output=True, text=True)

    pdf_file = os.path.join(report_dir, main_tex.replace(".tex", ".pdf"))
    if os.path.exists(pdf_file) and res2.returncode == 0:
        print(f"PDF successfully compiled: {pdf_file}")
        return True
    else:
        print(f"LaTeX compilation finished with code {res2.returncode}.")
        if os.path.exists(pdf_file):
            print(f"PDF was generated despite non-zero code: {pdf_file}")
            return True
        print("Compilation Log snippet:")
        log_file = os.path.join(report_dir, main_tex.replace(".tex", ".log"))
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                print("".join(lines[-30:]))
        return False


def sync_all(csv_path: str = "benchmark_results.csv", report_dir: str = "Report"):
    """Coordinates full update of tables, figures, and compiled PDF."""
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping table generation.")
        return

    table_path = os.path.join(report_dir, "tables", "results_table.tex")
    fig_dir = os.path.join(report_dir, "figures")

    generate_latex_table(csv_path, table_path)
    update_figures(csv_path, fig_dir)
    compile_latex(report_dir, "main.tex")


if __name__ == "__main__":
    sync_all()
