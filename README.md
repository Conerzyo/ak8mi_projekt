# Úkol do předmětu AK8MI 2022/2023
Python 3.10

Tento repozitář obsahuje kód pro obě zadané úlohy. Složky `backpack` a `benchmarking` obsahují kód python script, který
se snaží řešit zadané problémy. Složka `output` obsahuje výstupy programu.

## Řešené problémy
### _Benchmarking_ 

Porovnání **Random Search** a **Simulated Annealing**. Hodnotící funkce jsou DeJong 1, DeJong 2, Schwefel.
Program se spouští pro různá nastavení. Výsledky jsou zaznamenány do souboru `statistics.csv`. Zpracované grafy jsou 
ve složce `output` s pojmenováním obsahující algoritmus, testovací funkci, dimenze. Pro každou takovou kombinaci
je vygenerován graf průměrné hodnoty pro každý běh a konvergenční graf.

### _Knapsack problem_
Řešení 0-1 KP pomocí **brute force** a **simulovaného žíhání**. Výsledné grafy jsou zaznamenány do složky `output`.
Grafy znázorňují vývoj hodnoty batohu v průběhu běhu algoritmu.