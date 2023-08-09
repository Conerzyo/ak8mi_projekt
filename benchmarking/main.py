import os
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Definice testovacích funkcí
def dejong_1(x):
    return sum(xi ** 2 for xi in x)


def dejong_2(x):
    return sum((i + 1) * xi ** 2 for i, xi in enumerate(x))


def schwefel(x):
    return sum(-xi * np.sin(np.sqrt(abs(xi))) for xi in x)


# Algoritmy
def random_search(cost_function, dim, max_iterations):
    best_results = []
    for _ in range(max_iterations):
        solution = np.random.uniform(-5.12, 5.12, dim)
        cost = cost_function(solution)
        best_results.append(cost)
    return best_results


def simulated_annealing(cost_function, dim, max_iterations, max_temp=1000, min_temp=0.1, cooling_rate=0.95):
    best_results = []
    solution = np.random.uniform(-5.12, 5.12, dim)
    cost = cost_function(solution)
    temperature = max_temp

    for iteration in range(max_iterations):
        for _ in range(10):  # 10 FES per iteration
            new_solution = solution + np.random.normal(0, 1, dim)
            new_solution = np.clip(new_solution, -5.12, 5.12)
            new_cost = cost_function(new_solution)
            delta = new_cost - cost

            if delta < 0 or random.random() < np.exp(-delta / temperature):
                solution = new_solution
                cost = new_cost

        best_results.append(cost)
        temperature *= cooling_rate

        if temperature < min_temp:
            break

    # doplnění NaN hodnotami pro iterace, které se již neprovedly
    remaining_iterations = max_iterations - len(best_results)
    best_results += [np.nan] * remaining_iterations

    return best_results


# Spuštění algoritmů, statistické výstupy a generování grafů
if __name__ == '__main__':
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_folder, exist_ok=True)

    # Základní nastavení
    dimension_sizes = [5, 10]
    iterations = 1000
    runs = 30

    algorithms = {'Random Search': random_search, 'Simulované žíhání': simulated_annealing}
    functions = {'DeJong1': dejong_1, 'DeJong2': dejong_2, 'Schweffel': schwefel}

    statistics_data = []

    for algorithm_name, algorithm in algorithms.items():
        for function_name, function in functions.items():
            for current_dimensions_size in dimension_sizes:
                all_best_results = []
                all_best_results_without_nans = []
                for _ in range(runs):
                    best_results = algorithm(function, current_dimensions_size, iterations)
                    best_results_filtered = list(filter(lambda x: not np.isnan(x), best_results))
                    all_best_results.append(best_results)
                    all_best_results_without_nans.append(best_results_filtered)

                # Příprava dat pro grafy
                min_values = np.min(all_best_results, axis=0)
                max_values = np.max(all_best_results, axis=0)
                mean_values = np.mean(all_best_results, axis=0)
                median_values = np.median(all_best_results, axis=0)
                std_dev_values = np.std(all_best_results, axis=0)

                # Přípava dat pro uložení výsledků do csv
                min_values_csv = np.min(all_best_results_without_nans, axis=0)
                max_values_csv = np.max(all_best_results_without_nans, axis=0)
                mean_values_csv = np.mean(all_best_results_without_nans, axis=0)
                median_values_csv = np.median(all_best_results_without_nans, axis=0)
                std_dev_values_csv = np.std(all_best_results_without_nans, axis=0)

                statistics_data.append({'Algoritmus': algorithm_name,
                                        'Testovací funkce': function_name,
                                        'Dimenze': current_dimensions_size,
                                        'Minimum': np.min(min_values_csv),
                                        'Maximum': np.max(max_values_csv),
                                        'Průměr': np.mean(mean_values_csv),
                                        'Medián': np.median(median_values_csv),
                                        'Odchylka': np.mean(std_dev_values_csv)})

                # Konvergenční graf všech běhů
                plot_path = os.path.join(output_folder, f'{algorithm_name}_{function_name}_D{current_dimensions_size}_konvergence.png')
                plt.figure(figsize=(10, 6))
                for i in range(runs):
                    plt.plot(range(iterations), all_best_results[i], alpha=0.5)

                plt.xlabel('Iterace')
                plt.ylabel('Cost')
                plt.title(f'{algorithm_name} - {function_name} (D={current_dimensions_size}) - Konvergence')
                plt.grid(True)
                plt.savefig(plot_path)
                plt.close()
                print(f"[INFO] Kovergenční graf všechn běhů uložen v \"{plot_path}\"")

                # Průměrný konvergenční graf
                mean_plot_path = os.path.join(output_folder, f'{algorithm_name}_{function_name}_D{current_dimensions_size}_avg_konvergence.png')
                plt.figure(figsize=(10, 6))
                plt.plot(range(iterations), mean_values, label='Mean', color='blue')
                plt.fill_between(range(iterations),
                                 # znázornění rozptylu
                                 mean_values - std_dev_values,
                                 mean_values + std_dev_values,
                                 alpha=0.2,
                                 color='blue')
                plt.xlabel('Iterace')
                plt.ylabel('Cost')
                plt.title(f'{algorithm_name} - {function_name} (D={current_dimensions_size}) - Průměrná konvergence')
                plt.grid(True)
                plt.legend()
                plt.savefig(mean_plot_path)
                plt.close()
                print(f"[INFO] Graf průměrné konvergence uložen v \"{mean_plot_path}\"")

    # Uložeí statistických výstupů do tabulky
    csv_path = os.path.join(output_folder, 'statistics.csv')
    statistics_df = pd.DataFrame(statistics_data)
    statistics_df.to_csv(csv_path, index=False)
    print(f"[INFO] Statistická data uložena v \"{csv_path}\"")

    print(f"\n[INFO] Výstupní tabulka a grafy byly uloženy do složky \"{output_folder}\".")
    exit(0)