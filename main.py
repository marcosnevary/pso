algorithm_a = "PSOX (JAX)"
algorithm_b = "PSO (JAX)"

threshold = 0.05
df_filtered = df[df["algorithm"].isin([algorithm_a, algorithm_b])]

results = []
for benchmark in df_filtered["benchmark"].unique():
    for dimension in df_filtered["dimension"].unique():
        subset = df_filtered[
            (df_filtered["benchmark"] == benchmark)
            & (df_filtered["dimension"] == dimension)
        ]

        fitness_a = subset[subset["algorithm"] == algorithm_a][
            "best_fitness"
        ].to_numpy()
        fitness_b = subset[subset["algorithm"] == algorithm_b][
            "best_fitness"
        ].to_numpy()

        stat, p_value = stats.wilcoxon(fitness_a, fitness_b, alternative="two-sided")

        results.append(
            {
                "benchmark": benchmark,
                "dimension": dimension,
                f"p < {threshold}": p_value < threshold,
                "best_algorithm": algorithm_a
                if np.median(fitness_a) < np.median(fitness_b)
                else algorithm_b,
            },
        )

df_results = pd.DataFrame(results)
df_results
