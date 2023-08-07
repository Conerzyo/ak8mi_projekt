from random import random, randint, choices
from itertools import combinations
from time import time
import numpy as np
from matplotlib import pyplot as plt


def generate_instance(number_of_items):
    _items = []
    for i in range(number_of_items):
        _item = {
            "id": i,
            "volume": randint(1, 50),
            "value": randint(1, 50)
        }
        _items.append(_item)

    return _items


def set_capacity(number_of_items):
    if number_of_items <= 15:
        return 100
    elif number_of_items <= 30:
        return 200
    else:
        return 300


def calculate_combination_value(combination, items):
    total_value = 0
    total_volume = 0

    if isinstance(combination[0], dict):
        for idx in combination:
            id, _, _ = idx.values()
            total_value += items[id]["value"]
            total_volume += items[id]["volume"]
    elif isinstance(combination[0], int):
        for idx, val in enumerate(combination, start=0):
            if val == 1:
                total_value += items[idx]["value"]
                total_volume += items[idx]["volume"]

    return total_value, total_volume


def brute_force_solution(provided_items, maximum_capacity):
    all_runs_data = []
    _best_combination = []
    _best_value = 0
    start_time = time()

    for r in range(1, len(provided_items) + 1):
        for combination in combinations(provided_items, r):
            value, volume = calculate_combination_value(combination, provided_items)
            if volume <= maximum_capacity and value > _best_value:
                _best_value = value
                _best_combination = combination
            all_runs_data.append(value)

    end_time = time()
    return _best_combination, _best_value, end_time - start_time, all_runs_data


def generate_neighbour_solution(current_solution, number_of_items, maximum_capacity, max_attempts=10):
    for _ in range(max_attempts):
        neighbour_solution = current_solution.copy()
        invert_probability = 1 / number_of_items

        for idx in range(number_of_items):
            if random() < invert_probability:
                neighbour_solution[idx] = 1 - neighbour_solution[idx]

        neighbour_value, neighbour_volume = calculate_combination_value(neighbour_solution, items)

        if neighbour_volume <= maximum_capacity:
            return neighbour_solution

    # Pokud počet pokusů překročí max_attempts, vrať aktuální řešení.
    return current_solution


def simulated_annealing_solution(provided_items, maximum_capacity, fes, max_temperature, min_temperature, cooling_rate = 0.999):
    _num_items = len(provided_items)
    _current_solution = choices([0, 1], k=_num_items)
    _current_temperature = max_temperature
    _current_iteration = 1

    best_volume = 0
    best_solution = _current_solution.copy()
    best_value = 0
    temperature_list = []
    best_values_list = []
    
    while _current_temperature > min_temperature and _current_iteration < fes:
        # vygenerování sousedního řešení
        _neighbour_solution = generate_neighbour_solution(_current_solution, _num_items, maximum_capacity)

        # ohodnocení aktuálního a sousedního řešení
        _current_value, _current_volume = calculate_combination_value(_current_solution, provided_items)
        _neighbour_value, _neighbour_volume = calculate_combination_value(_neighbour_solution, provided_items)

        _delta = _neighbour_value - _current_value

        if _delta > 0 or random() < np.exp(_delta / _current_temperature):
            if _neighbour_volume <= maximum_capacity:
                best_solution = _neighbour_solution.copy()
                best_value = _neighbour_value
                best_volume = _neighbour_volume
            else:
                best_solution = _current_solution.copy()
                best_value = _current_value
                best_volume = _current_volume

        temperature_list.append(_current_temperature)
        best_values_list.append(best_value)

        _current_temperature *= cooling_rate
        _current_iteration += 1

    return best_solution, best_value, best_volume, temperature_list, best_values_list


def print_items(desired_items, header_text="Vygenerované předměty", print_summary=False):
    if desired_items is None or len(desired_items) == 0:
        print("Žádné předměty k vypsání")
        return

    print(f"{header_text}:")
    print("=================================")
    print("|", "Id", "|", "Objem", "|", "Hodnota", "|", sep="\t")
    _sum_volume = 0
    _sum_value = 0
    for item in desired_items:
        _sum_volume += item["volume"] if isinstance(item, dict) else item
        _sum_value += item["value"] if isinstance(item, dict) else item
        print("|", item["id"], "|", item["volume"], "\t|", item["value"], "\t|", sep="\t")

    if print_summary:
        print("|-------------------------------|")
        print("|", "  ", "|", _sum_volume, "\t|", _sum_value, "\t|", sep="\t")
    print("=================================\n\n")


def gather_items_from_solution(best_solution, items):
    if best_solution is None or len(best_solution) == 0:
        return []

    _result = []

    for idx, val in enumerate(best_solution, start=0):
        if val == 1:
            _result.append(items[idx])

    return _result


if __name__ == "__main__":
    num_items = 50

    items = generate_instance(num_items)
    capacity = set_capacity(num_items)

    print_items(items)

    # brute force řešení
    print("Provádím brute force řešení...")
    best_combination, best_value, execution_time, all_runs_data = brute_force_solution(items, capacity)

    # Výpis nalezeného řešení
    print(f"Celková hodnota: {best_value}")
    print(f"Čas výpočtu: {execution_time:.2f} s")
    print_items(best_combination, "Nejlepší kombinace", True)

    # Statistiky pro brute force řešení
    mean_per_iteration_brute_force = np.mean(all_runs_data)
    std_dev_per_iteration_brute_force = np.std(all_runs_data)

    # Vytvoření konvergenčního grafu pro brute force řešení
    plt.figure()
    plt.plot(range(len(all_runs_data)), all_runs_data, label='Hodnota')
    plt.axhline(y=mean_per_iteration_brute_force, color='red', linestyle='--', label='Průměr')
    plt.axhline(y=mean_per_iteration_brute_force + std_dev_per_iteration_brute_force, color='green', linestyle=':',
                label='Horní mez konf. int.')
    plt.axhline(y=mean_per_iteration_brute_force - std_dev_per_iteration_brute_force, color='green', linestyle=':',
                label='Spodní mez konf. int.')
    plt.xlabel("Iterace")
    plt.ylabel("Hodnota nejlepší kombinace")
    plt.title("Konvergenční graf Brute Force řešení")
    plt.legend()
    plt.savefig("output/konvergencni_graf_brute_force_kp.png")

    # simulované žíhání
    print("Provádím simulované žíhání...")

    # parametry
    max_temp = 1000
    min_temp = 0.1
    fes = (2**num_items) - 1   # 2^n - 1 (počet všech možných kombinací)
    num_runs = 30
    cooling_rate = 0.995
    all_best_values_sa = []

    overall_best_solution = None
    overall_best_value = 0
    overall_best_temp_list = None
    overall_best_value_list = None

    for i in range(num_runs):
        best_solution_run, best_value_run, best_volume_run, temp_list_run, value_list_run \
            = simulated_annealing_solution(items, capacity, fes, max_temp, min_temp, cooling_rate)

        print(f"Simulované žíhání - běh {i + 1}/{num_runs} - hodnota: {best_value_run} - objem: {best_volume_run} - zvolené předměty: {best_solution_run}")
        if best_value_run > overall_best_value and best_volume_run <= capacity:
            print(f"Nová nejlepší hodnota: {best_value_run} s objemem: {best_volume_run}")
            overall_best_solution = best_solution_run
            overall_best_value = best_value_run
            overall_best_temp_list = temp_list_run
            overall_best_value_list = value_list_run
        all_best_values_sa.append(value_list_run)

    # Výpis nalezeného řešení pro simulované žíhání
    print(f"Celková hodnota (simulované žíhání): {overall_best_value}")

    solution_items_from_sa = gather_items_from_solution(overall_best_solution, items)
    print_items(solution_items_from_sa, "Nejlepší kombinace předmětů (simulované žíhání)", True)

    # Vytvoření konvergenčního grafu a statistik pro simulované žíhání
    if overall_best_temp_list is not None or overall_best_value_list is not None:
        mean_per_iteration = np.mean(all_best_values_sa, axis=0)
        std_dev_per_iteration = np.std(all_best_values_sa, axis=0)

        # Vytvoření konvergenčního grafu s rozptylem
        plt.figure()
        plt.plot(range(len(overall_best_value_list)), mean_per_iteration, label='Průměr')
        plt.fill_between(range(len(mean_per_iteration)),
                         # Vykreslení rozptylu
                         mean_per_iteration - std_dev_per_iteration,
                         mean_per_iteration + std_dev_per_iteration, alpha=0.2, label='Rozptyl')
        plt.xlabel("Iterace")
        plt.ylabel("Hodnota nejlepší kombinace")
        plt.title("Konvergenční graf Simulovaného žíhání s rozptylem")
        plt.legend()
        plt.savefig("output/konvergencni_graf_sa_kp_s_rozptylem.png")

    # Konec programu
    exit(0)
