import ast
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save_figure(fig: plt.Figure, filename: str, config: dict) -> None:
    save_path = config["output_path"] / filename
    fig.savefig(save_path, bbox_inches="tight", format="pdf", dpi=300)
    plt.close(fig)


def plot_execution_time(df: pd.DataFrame, config: dict) -> None:
    benchmarks = df["Benchmark"].unique()
    algorithms = df["Algorithm"].unique()
    dimensions = df["Dimension"].unique()

    colors = ["#909090", "#0080FF"]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        },
    )

    fig, axes = plt.subplots(2, 2, figsize=(5, 4), sharex=True, sharey=True)
    axes_flattened = axes.flatten()

    lines = []
    labels = []

    for idx_ax, (ax, benchmark) in enumerate(zip(axes_flattened, benchmarks)):
        df_benchmark = df[df["Benchmark"] == benchmark]

        for idx_algo, algorithm in enumerate(algorithms):
            df_subset = df_benchmark[df_benchmark["Algorithm"] == algorithm]
            mean = df_subset["Mean of Execution Times (s)"]
            std = df_subset["Standard Deviation of Execution Times (s)"]

            (line,) = ax.plot(
                dimensions,
                mean,
                label=algorithm,
                marker="o",
                color=colors[idx_algo],
                markersize=6,
                linewidth=2,
            )
            ax.fill_between(dimensions, mean - std, mean + std, color=colors[idx_algo], alpha=0.2)

            if idx_ax == 0:
                lines.append(line)
                labels.append(algorithm)

        ax.set_yscale("log")
        ax.set_title(f"{benchmark}")
        sns.despine(ax=ax, left=True)

    fig.supxlabel("Dimensão")
    fig.supylabel("Tempo de Execução (s)")

    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(algorithms),
        frameon=False,
    )

    plt.tight_layout()
    _save_figure(fig, "execution_time_plot.pdf", config)


def plot_convergence(df: pd.DataFrame, config: dict) -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        },
    )

    benchmarks = df["Benchmark"].unique()
    dimensions = df["Dimension"].unique()
    algorithms = df["Algorithm"].unique()

    colors = ["#909090", "#0080FF"]

    fig, axes = plt.subplots(4, 4, figsize=(12, 12), sharex=True)

    lines = []
    labels = []

    for i, benchmark in enumerate(benchmarks):
        for j, dimension in enumerate(dimensions):
            ax = axes[i, j]
            df_subset = df[(df["Dimension"] == dimension) & (df["Benchmark"] == benchmark)]

            for k, (_, row) in enumerate(df_subset.iterrows()):
                mean_history = jnp.array(row["Mean Fitness History"])
                std_history = jnp.array(row["Std Fitness History"])
                iterations = range(len(mean_history))

                (line,) = ax.plot(
                    iterations,
                    mean_history,
                    label=row["Algorithm"],
                    color=colors[k],
                    linewidth=2,
                )
                ax.fill_between(
                    iterations,
                    mean_history - std_history,
                    mean_history + std_history,
                    alpha=0.2,
                    color=colors[k],
                )

                if i == 0 and j == 0:
                    lines.append(line)
                    labels.append(row["Algorithm"])

            sns.despine(ax=ax, left=True)
            ax.set_title(f"{benchmark} (d = {dimension})")

    fig.supxlabel("Iteração")
    fig.supylabel("Fitness")

    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=len(algorithms),
        frameon=False,
    )

    plt.tight_layout()
    _save_figure(fig, "convergence_plot.pdf", config)


def _generate_comparison_table(
    df: pd.DataFrame,
    config: dict,
    mean_col: str,
    std_col: str,
    output_filename: str,
    caption: str,
    label: str,
) -> None:
    df_proc = df.copy()

    df_proc["formatted"] = (
        df_proc[mean_col].map("{:.2e}".format) + r" $\pm$ " + df_proc[std_col].map("{:.2e}".format)
    )

    df_pivot = df_proc.pivot_table(
        index=["Benchmark", "Dimension"],
        columns="Algorithm",
        values=["formatted", mean_col],
        aggfunc="first",
    )

    means = df_pivot[mean_col].fillna(float("inf"))

    display_df = df_pivot["formatted"].copy()

    for index, row in means.iterrows():
        min_val = row.min()
        is_min = row == min_val
        for col in display_df.columns:
            if is_min[col]:
                display_df.loc[index, col] = f"\\textbf{{{display_df.loc[index, col]}}}"

    display_df = display_df.reset_index()

    output_path = config["output_path"] / output_filename

    latex_code = display_df.style.hide(axis="index").to_latex(
        column_format="llcc",
        hrules=True,
        caption=caption,
        label=label,
        position="h",
    )

    with output_path.open("w") as f:
        f.write(latex_code)


def create_convergence_table(df: pd.DataFrame, config: dict) -> None:
    _generate_comparison_table(
        df=df,
        config=config,
        mean_col="Mean of Fitness",
        std_col="Standard Deviation of Fitness",
        output_filename="convergence_table.tex",
        caption=r"Convergence comparison (Mean Fitness $\pm$ Std Dev). Best results in bold.",
        label="tab:convergence",
    )


def create_execution_time_table(df: pd.DataFrame, config: dict) -> None:
    _generate_comparison_table(
        df=df,
        config=config,
        mean_col="Mean of Execution Times (s)",
        std_col="Standard Deviation of Execution Times (s)",
        output_filename="execution_time_table.tex",
        caption="Execution time comparison in seconds.",
        label="tab:execution_time",
    )


def generate_visualizations(df: pd.DataFrame, config: dict) -> None:
    print("Plotting convergence...")
    plot_convergence(df, config)

    print("Plotting execution time...")
    plot_execution_time(df, config)

    print("Creating convergence table (LaTeX)...")
    create_convergence_table(df, config)

    print("Creating execution time table (LaTeX)...")
    create_execution_time_table(df, config)


config = {
    "output_path": Path("./results/"),
    "palette": "Blues",
}


if __name__ == "__main__":
    results = pd.read_csv(config["output_path"] / "benchmark_results.csv")
    results["Execution Time History"] = results["Execution Time History"].apply(ast.literal_eval)
    results["Mean Fitness History"] = results["Mean Fitness History"].apply(ast.literal_eval)
    results["Std Fitness History"] = results["Std Fitness History"].apply(ast.literal_eval)
    generate_visualizations(results, config)
