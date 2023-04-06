import json
from typing import Dict
from nptyping import Float, NDArray, Shape, Int
import numpy as np
import seaborn
import pandas as pd
import matplotlib.pyplot as plt

from src.V2.Functions.reshape import reshape_one, reshape_order
from src.V2.Functions.kendall_distance import kendall_matrix_distance


output_files = ["data/fat_grid_2.json"]

results = []

for filename in output_files:
    with open(filename, 'r') as the_file:
        results += json.loads(the_file.read())
        # results.append(json.loads(the_file.read()))


# Tri des résultats
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

# q_learn = list(filter(filterer({
#     "algorithm": 'Q_learning'
#     }), results))
# q_learn.sort(reverse=True, key = lambda element: element["epoch_number"])

# best_success = list(filter(filterer({
#     "algorithm": 'Q_learning',
#     "success": 0.65
#     }), results))
# len(best_success)


# distance_matrix = np.array(distance_matrix, float)
# dataframe = pd.DataFrame(distance_matrix)
# seaborn.heatmap(dataframe, fmt=".1f", annot=True)

# # Avec la matrix d'ordre
# distance_matrix = np.full((len(best_success),len(best_success)), float)
# for row_index in range(len(best_success)):
#     for col_index in range(row_index, len(best_success)):
#         distance = kendall_matrix_distance(best_success[row_index]["reshape_order"], best_success[col_index]["reshape_order"])
#         distance_matrix[row_index, col_index] = distance
#         distance_matrix[col_index, row_index] = distance

# distance_matrix = np.array(distance_matrix, float)
# dataframe = pd.DataFrame(distance_matrix)
# seaborn.heatmap(dataframe, fmt=".1f", annot=True)


# # Mieux investiguer les meilleurs solutions
# import random
# import gymnasium as gym

# from src.V2.Classes.Policy import Policy
# from src.V2.Classes.Agent import Agent
# from src.V2.Functions.run import run_static

# seed = 10
# test_iteration = 5000
# desc=["SFFF", "FHFH", "FFFH", "HFFG"] # Same as the map called "4*4"
# environment = gym.make('FrozenLake-v1', desc=desc, is_slippery=True, render_mode="rgb_array")

# for solution in best_success:
#     success = 0
#     random.seed(seed)
#     environment.reset(seed=seed)
#     deterministic_policy = Policy.buildOptimalPolicyFrom(np.asarray(solution["q_sa"]))
#     for epoch in range(test_iteration):
#         test_agent = Agent(deterministic_policy)
#         run_static(environment = environment, agent = test_agent)
#         success += test_agent.current_state_index == (environment.observation_space.n - 1) # type: ignore
#     solution["extended_test"] = success / test_iteration

# best_success.sort(reverse=True, key = lambda element: element["extended_test"])

# Plot du succes en fonction des paramètres

big_df = pd.DataFrame(results)
# big_df.melt(id_vars=['alpha', 'gamma', 'algorithm', 'epsilon'], value_vars=['epoch_number', 'success'])
# seaborn.lineplot(big_df.melt(id_vars=['alpha', 'gamma', 'algorithm', 'epsilon'], value_vars=['epoch_number', 'success']), x='epoch_number', y='success', hue='algorithm')
# from scipy import stats
# target_df = pd.DataFrame()
# target_list = []
# sub_df = big_df[['alpha', 'gamma', 'algorithm', 'epsilon', 'success', 'epoch_number']]
# # See https://stackoverflow.com/a/60909312
# cols = ['algorithm', 'alpha', 'gamma', 'epsilon']
# for k, d in sub_df.drop(cols, axis=1).groupby([sub_df[c] for c in cols]):
#     out = d
#     out["algorithm"] = k[0]
#     target_list.append(out)
#     print(np.corrcoef(out["epoch_number"], out["success"])[0, 1])
#     print(stats.pearsonr(out["epoch_number"], out["success"]))
# fig, ax = plt.subplots()
# for line in target_list[0:100]:
#     seaborn.lineplot(line, x='epoch_number', y='success', hue='algorithm', ax = ax)
# seaborn.lineplot(out, x='epoch_number', y='success', hue='algorithm', ax = ax)
# plt.show()
# seaborn.lineplot(target_df.melt(id_vars=['epoch_number', 'index', 'algorithm']), x='epoch_number', y='value', hue='algorithm')



plot = seaborn.boxplot(data=big_df, x="epoch_number", y="success", hue="algorithm").tick_params(axis='x', rotation=90)
plot.tick_params(axis='x', rotation=90)
plot.figure

seaborn.boxplot(big_df, x = "epsilon", y= "success", hue='algorithm').tick_params(axis='x', rotation=90)

seaborn.boxplot(big_df, x = "alpha", y= "success", hue='algorithm').tick_params(axis='x', rotation=90)

seaborn.boxplot(big_df, x = "gamma", y= "success", hue='algorithm').tick_params(axis='x', rotation=90)

seaborn.boxplot(big_df, x = "epoch_number", y= "success", hue='algorithm').tick_params(axis='x', rotation=90)

seaborn.pairplot(big_df, hue='algorithm', height=2.5)

## Meiux voir les densités
#https://seaborn.pydata.org/tutorial/distributions.html

# Avec un peu de filtre

sub_results = list(filter(
    lambda element: 
        element["success"] > 0.5 and element["algorithm"] != "MC",
        results
))

better_df = pd.DataFrame(sub_results)

target_df = pd.DataFrame()
target_list = []
sub_df = better_df[['alpha', 'gamma', 'algorithm', 'epsilon', 'success', 'epoch_number']]
# See https://stackoverflow.com/a/60909312
cols = ['algorithm', 'alpha', 'gamma', 'epsilon']
for k, d in sub_df.drop(cols, axis=1).groupby([sub_df[c] for c in cols]):
    out = d
    out["algorithm"] = k[0]
    target_list.append(out)

fig, ax = plt.subplots()
for line in target_list[0:100]:
    seaborn.lineplot(line, x='epoch_number', y='success', ax = ax)



better_df.filter(like = "SARSA", axis = 1)
sum(better_df["algorithm"=="SARSA"])
seaborn.lineplot(better_df, x = "epoch_number", y= "success", hue='algorithm')

seaborn.jointplot(better_df, x = "epoch_number", y= "success", kind='kde')

# On veut voir si ces solutions sont "semblables"

# Avec la matrix du plus grand élément (meilleur)
distance_matrix = np.full((len(big_df),len(big_df)), float)
for row_index in range(len(big_df)):
    for col_index in range(row_index, len(big_df)):
        distance = kendall_matrix_distance(big_df["reshape_one"][row_index], big_df["reshape_one"][col_index])
        distance_matrix[row_index, col_index] = distance
        distance_matrix[col_index, row_index] = distance
distance_matrix = np.array(distance_matrix, float)
dataframe = pd.DataFrame(distance_matrix)
seaborn.heatmap(dataframe)
