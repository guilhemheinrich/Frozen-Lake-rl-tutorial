import json
from typing import Dict
from nptyping import Float, NDArray, Shape, Int
import numpy as np
import seaborn
import pandas as pd
import matplotlib.pyplot as plt

from src.V2.Functions.reshape import reshape_one, reshape_order
from src.V2.Functions.kendall_distance import kendall_matrix_distance


output_file = "data/fat_grid_2.json"

with open(output_file, 'r') as the_file:
    results: list = json.loads(the_file.read())


# Tri des résultas
results.sort(reverse=True, key = lambda element: element["success"])

# Ajout de "signature de matrice" pour mieux comparer les Q_sa, et ainsi étudier la convergeance
for result in results:
    result["reshape_one"] = reshape_one(np.asarray(result["q_sa"]))
    result["reshape_order"] = reshape_order(np.asarray(result["q_sa"]))

# Une fonction pour contruire des filtres
def filterer(options: Dict):
    def filter(element: Dict):
        for key in options:
            if element[key] != options[key]:
                return False
        return True
    return filter

q_learn = list(filter(filterer({
    "algorithm": 'Q_learning'
    }), results))
q_learn.sort(reverse=True, key = lambda element: element["epoch_number"])

best_success = list(filter(filterer({
    "algorithm": 'Q_learning',
    "success": 0.65
    }), results))
len(best_success)

# On veut voir si ces solutions sont "semblables"

# Avec la matrix du plus grand élément (meilleur)
distance_matrix = np.full((len(best_success),len(best_success)), float)
for row_index in range(len(best_success)):
    for col_index in range(row_index, len(best_success)):
        distance = kendall_matrix_distance(best_success[row_index]["reshape_one"], best_success[col_index]["reshape_one"])
        distance_matrix[row_index, col_index] = distance
        distance_matrix[col_index, row_index] = distance

distance_matrix = np.array(distance_matrix, float)
dataframe = pd.DataFrame(distance_matrix)
seaborn.heatmap(dataframe, fmt=".1f", annot=True)

# Avec la matrix d'ordre
distance_matrix = np.full((len(best_success),len(best_success)), float)
for row_index in range(len(best_success)):
    for col_index in range(row_index, len(best_success)):
        distance = kendall_matrix_distance(best_success[row_index]["reshape_order"], best_success[col_index]["reshape_order"])
        distance_matrix[row_index, col_index] = distance
        distance_matrix[col_index, row_index] = distance

distance_matrix = np.array(distance_matrix, float)
dataframe = pd.DataFrame(distance_matrix)
seaborn.heatmap(dataframe, fmt=".1f", annot=True)


# Mieux investiguer les meilleurs solutions
import random
import gymnasium as gym

from src.V2.Classes.Policy import Policy
from src.V2.Classes.Agent import Agent
from src.V2.Functions.run import run_static

seed = 10
test_iteration = 5000
desc=["SFFF", "FHFH", "FFFH", "HFFG"] # Same as the map called "4*4"
environment = gym.make('FrozenLake-v1', desc=desc, is_slippery=True, render_mode="rgb_array")

for solution in best_success:
    success = 0
    random.seed(seed)
    environment.reset(seed=seed)
    deterministic_policy = Policy.buildOptimalPolicyFrom(np.asarray(solution["q_sa"]))
    for epoch in range(test_iteration):
        test_agent = Agent(deterministic_policy)
        run_static(environment = environment, agent = test_agent)
        success += test_agent.current_state_index == (environment.observation_space.n - 1) # type: ignore
    solution["extended_test"] = success / test_iteration

best_success.sort(reverse=True, key = lambda element: element["extended_test"])

# Plot du succes en fonction des paramètres

big_df = pd.DataFrame(best_success)
seaborn.boxplot(data=big_df, x="epoch_number", y="extended_test", hue="algorithm")





