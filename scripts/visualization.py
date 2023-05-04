import numpy as np
import sys
import os
from pathlib import Path
from mprnn.utils import FILEPATH
import matplotlib.pyplot as plt


trainiters = [*[10000]*796]
totaliters = 0
np_dict = {}
for it in trainiters:
    totaliters += it
    os.chdir(Path(FILEPATH, "data", "models", "run0", "notSSP",f"{ totaliters}","data"))
    filename = "match_history.npy"
    try:
        np_dict.update(np.load(filename, allow_pickle=True).item())
        print("Loaded in data from: " + str(totaliters))
    except FileNotFoundError:
        print(f"Error: Numpy '{filename}' not found.")
    except Exception as e:
        print(f"Error: {e}")

def gen_plot(game_data, opponent):
    correct = 0
    incorrect = 0
    ratio = []

    for step in game_data:
        agent_choice = game_data[step][0]
        opponent_choice = game_data[step][1]

        if agent_choice == opponent_choice:
                correct += 1
        else:
                incorrect += 1
        
        ratio.append(correct / (correct + incorrect))

    del_key = list(range(1, 100001))
    for key in del_key:
        del game_data[key]

    plt.plot(list(game_data.keys()), ratio[100000:])
    plt.title(f'MP Performance {opponent}')
    plt.xlabel('Step')
    plt.ylabel('Matching Ratio')
    plt.savefig(f"performance_{opponent}")


os.chdir(Path(FILEPATH))
try:
    game_data = np_dict
    opponent = "Influence_3_15_75"
    gen_plot(game_data, opponent)
except Exception as e:
    print(f"Error: {e}")
