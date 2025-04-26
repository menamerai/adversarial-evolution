#!/usr/bin/env python3
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Plot a heatmap from a round‑robin compare_results CSV"
    )
    parser.add_argument(
        "csv_path", help="Path to compare_results.csv (Agent_A,Agent_B,A_avg,…)"
    )
    parser.add_argument(
        "--metric",
        choices=["A_avg","B_avg","A_std","B_std","composite"],
        default="A_avg",
        help="Which column to display in the heatmap",
    )
    parser.add_argument(
        "--output",
        default="compare_results_heatmap.png",
        help="Where to save the PNG plot",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(8,6),
        help="Figure size, e.g. --figsize 10 8",
    )
    parser.add_argument(
        "--cmap",
        default="coolwarm",
        help="Matplotlib colormap name",
    )
    parser.add_argument(
        "--annot",
        action="store_true",
        help="Annotate cells with values",
    )
    args = parser.parse_args()

    # load and pivot
    df = pd.read_csv(args.csv_path)
    # compute composite metric as sum of average scores
    df["composite"] = df["A_avg"] + df["B_avg"]
    pivot = df.pivot(index="Agent_A", columns="Agent_B", values=args.metric)

    # plot
    plt.figure(figsize=tuple(args.figsize))
    sns.heatmap(
        pivot,
        annot=args.annot,
        fmt=".2f",
        cmap=args.cmap,
        center=0,
        cbar_kws={"label": args.metric},
    )
    plt.title(f"{args.metric} heatmap")
    plt.xlabel("Agent_B")
    plt.ylabel("Agent_A")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Heatmap saved to {args.output}")

if __name__ == "__main__":
    main()