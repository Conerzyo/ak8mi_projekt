from random import random, randint
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


def generate_backpack_with_n_picked(number_of_items):
    arr = np.zeros(number_of_items, dtype=int)
    ones = min(3, number_of_items)
    arr[:ones] = 1
    np.random.shuffle(arr)
    return arr


def set_capacity(number_of_items):
    if number_of_items <= 15:
        return 100
    elif number_of_items <= 30:
        return 200
    else:
        return 300


def calculate_combination_value(combination, items, bit_picking=True):
    total_value = 0
    total_volume = 0

    if bit_picking:
        for idx, val in enumerate(combination, start=0):
            if val == 1:
                total_value += items[idx]["value"]
                total_volume += items[idx]["volume"]
    else:
        for idx in combination:
            total_value += items[idx]["value"]
            total_volume += items[idx]["volume"]

    return total_value, total_volume

# Toto je velmi pomalé řešení, které jsem zkusil jako první. O(2^N)
# def brute_force_solution(provided_items, maximum_capacity):
#     _all_runs_data = []
#     _best_combination = []
#     _best_value = 0
#     _item_ids = [item["id"] for item in provided_items]
#     start_time = time()
#
#     for r in range(1, len(_item_ids) + 1):
#         for combination in combinations(_item_ids, r):
#             value, volume = calculate_combination_value(combination, provided_items, False)
#             if volume <= maximum_capacity and value > _best_value:
#                 _best_value = value
#                 _best_combination = combination
#             _all_runs_data.append(value)
#
#     end_time = time()
#     return _best_combination, _best_value, end_time - start_time, _all_runs_data


# 2D vektor pro dynamické programování O(N * V)
def brute_force_solution(provided_items, maximum_capacity):
    start_time = time()
    _all_runs_data = []
    _number_of_items = len(provided_items)

    matrix_id_volume = np.zeros((_number_of_items + 1, maximum_capacity + 1), dtype=int).tolist()

    for idx in range(1, _number_of_items + 1):
        item = provided_items[idx - 1]
        for v in range(1, maximum_capacity + 1):
            if item["volume"] <= v:
                matrix_id_volume[idx][v] = max(matrix_id_volume[idx - 1][v],
                                               matrix_id_volume[idx - 1][v - item["volume"]] + item["value"])
            else:
                matrix_id_volume[idx][v] = matrix_id_volume[idx - 1][v]

        _all_runs_data.append(matrix_id_volume[idx][maximum_capacity])

    _best_value = matrix_id_volume[_number_of_items][maximum_capacity]
    _best_combination = []
    idx, v = _number_of_items, maximum_capacity
    _step = 1
    print("\tProcházím matici a vybírám předměty")
    while idx > 0 and v > 0:
        item = provided_items[idx - 1]
        if matrix_id_volume[idx][v] != matrix_id_volume[idx - 1][v]:
            _best_combination.append(item["id"])
            v -= item["volume"]
            print(f"{_step}\tVybrán předmět s ID {item['id']}, objemem {item['volume']} a hodnotou {item['value']}. Zbývající kapacita batohu: {v}")
        else:
            print(f"{_step}\tPředmět s ID {item['id']} nebyl vybrán")
        idx -= 1
        _step += 1

    end_time = time()
    _best_combination.sort()

    print(f"\nDoba výpočtu: {end_time - start_time:.2f} s\n")
    print(f"Nejvyšší hodnota dosažená pomocí brute force: {_best_value}\n")
    return _best_combination, _all_runs_data


def generate_neighbour_solution(current_solution, items, number_of_items, maximum_capacity, bits_to_invert=1):
    neighbour_solution = current_solution.copy()

    for _ in range(bits_to_invert):
        idx = randint(0, number_of_items - 1)
        neighbour_solution[idx] = 1 - neighbour_solution[idx]

    neighbour_value, neighbour_volume = calculate_combination_value(neighbour_solution, items)

    if neighbour_volume <= maximum_capacity:
        return neighbour_solution

    return current_solution


def simulated_annealing_solution(provided_items, maximum_capacity, fes, max_temperature, min_temperature, cooling_rate=0.999):
    _num_items = len(provided_items)
    _current_solution = generate_backpack_with_n_picked(_num_items)
    _current_temperature = max_temperature
    _current_iteration = 1

    best_volume = 0
    best_solution = _current_solution.copy()
    best_value = 0
    temperature_list = []
    best_values_list = []
    
    while _current_temperature > min_temperature and _current_iteration < fes:
        # vygenerování sousedního řešení
        _neighbour_solution = generate_neighbour_solution(_current_solution,
                                                          provided_items,
                                                          _num_items,
                                                          maximum_capacity,
                                                          3 if _current_iteration < fes // 4 else 1)

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
        print("[ERROR] Žádné předměty k vypsání")
        return

    print(f"\t{header_text}:")
    print("\t=================================")
    print("\t|", "Id", "|", "Objem", "|", "Hodnota", "|", sep="\t")
    _sum_volume = 0
    _sum_value = 0
    for item in desired_items:
        _sum_volume += item["volume"] if isinstance(item, dict) else item
        _sum_value += item["value"] if isinstance(item, dict) else item
        print("\t|", item["id"], "|", item["volume"], "\t|", item["value"], "\t|", sep="\t")

    if print_summary:
        print("\t|-------------------------------|")
        print("\t|", "  ", "|", _sum_volume, "\t|", _sum_value, "\t|", sep="\t")
    print("\t=================================\n")


def gather_items_from_solution(best_solution, items, bit_picking=True):
    if best_solution is None or len(best_solution) == 0:
        return []

    _result = []

    if bit_picking:
        for idx, val in enumerate(best_solution, start=0):
            if val == 1:
                _result.append(items[idx])
    else:
        for idx in best_solution:
            _result.append(items[idx])

    return _result


def gather_ids_from_solution(best_solution):
    if best_solution is None or len(best_solution) == 0:
        return []

    _result = []
    for idx, val in enumerate(best_solution, start=0):
        if val == 1:
            _result.append(idx)

    return _result


if __name__ == "__main__":
    number_of_items = 15

    items = generate_instance(number_of_items)
    maximum_capacity = set_capacity(number_of_items)
    print(f"Na základě počtu předmětů byla vybrána kapacita batohu: {maximum_capacity}\n")
    print_items(items)

    # brute force řešení
    print("Zahajuji brute force řešení...")
    print("==============================\n")
    best_combination, all_runs_data = brute_force_solution(items, maximum_capacity)
    solution_items_from_bt = gather_items_from_solution(best_combination, items, False)

    print_items(solution_items_from_bt, "Nejlepší kombinace předmětů (brute force)", True)

    # Statistiky pro brute force řešení
    mean_per_iteration_brute_force = np.mean(all_runs_data)
    std_dev_per_iteration_brute_force = np.std(all_runs_data)

    # Vytvoření konvergenčního grafu pro brute force řešení
    plt.figure()
    plt.plot(range(len(all_runs_data)), all_runs_data, label='Hodnota')
    plt.axhline(y=mean_per_iteration_brute_force + std_dev_per_iteration_brute_force, color='green', linestyle=':',
                label='Horní mez konf. int.')
    plt.axhline(y=mean_per_iteration_brute_force - std_dev_per_iteration_brute_force, color='green', linestyle=':',
                label='Spodní mez konf. int.')
    plt.xlabel("Iterace")
    plt.ylabel("Hodnota nejlepší kombinace")
    plt.title("Konvergenční graf Brute Force řešení")
    plt.legend()
    plt.savefig("output/konvergencni_graf_brute_force_kp.png")
    print("[INFO] Graf byl uložen do složky ./output s názvem \"konvergencni_graf_brute_force_kp.png\"\n")

    # simulované žíhání
    print("Zahajuji řešení pomocí simulovaného žíhání...")
    print("=============================================\n")

    # parametry
    max_temp = 2000
    min_temp = 0.1
    # Maximální počet iterací pro O(2^N)
    # max_fes = (2 ** num_items) - 1   # 2^n - 1 (počet všech možných kombinací)

    # Maximální počet iterací pro O(N * V) - dynamické programování (2D vektor)
    max_fes = (number_of_items + 1) * (maximum_capacity + 1) - 1  # (n + 1) * (v + 1) - počet všech možných kombinací
    num_runs = 30
    cooling_rate = 0.999
    all_best_values_sa = []

    overall_best_solution = None
    overall_best_value = 0
    overall_best_temp_list = None
    overall_best_value_list = None

    for i in range(num_runs):
        best_solution_run, best_value_run, best_volume_run, temp_list_run, value_list_run \
            = simulated_annealing_solution(items, maximum_capacity, max_fes, max_temp, min_temp, cooling_rate)
        log_message = f"[Běh {i + 1}/{num_runs}] Simulované žíhání - hodnota: {best_value_run} - objem: {best_volume_run} - zvolené předměty: {gather_ids_from_solution(best_solution_run)}"

        if best_value_run > overall_best_value and best_volume_run <= maximum_capacity:
            overall_best_solution = best_solution_run
            overall_best_value = best_value_run
            overall_best_temp_list = temp_list_run
            overall_best_value_list = value_list_run
            log_message += f" <= Nové nejlepší řešení!"
        all_best_values_sa.append(value_list_run)
        print(log_message)

    # Výpis nalezeného řešení pro simulované žíhání
    print(f"\nNejvyšší hodnota dosažená pomocí simulovaného žíhání: {overall_best_value}\n")

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
        print(f"[INFO] Konvergenční graf {num_runs} běhů byl uložen do složky ./output s názvem \"konvergencni_graf_sa_kp_s_rozptylem.png\"\n")

    # Konec programu
    exit(0)
