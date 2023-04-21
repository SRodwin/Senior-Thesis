import os
from pathlib import Path
from mprnn.utils import FILEPATH

import numpy as np


STORE_STEPS = 10000
SAVE_STEPS = 2000000


class AbstractDataCollector():

    def __init__(self):
        self.init_data_vars()

    def init_data_vars(self):
        raise NotImplementedError


class DataCollector(AbstractDataCollector):
    def init_data_vars(self):
        print("Initalized DataCollector")
        self.match_snapshot = {}

    def save_snapshot(self, step, agent_choice, opponent_choice, history, pr):
        snap = [agent_choice, opponent_choice, history, pr]
        self.match_snapshot[step] = snap

    def check_save(self, opponent_action, action, current_steps, total_steps, history, pr):
        self.save_snapshot(total_steps, action, opponent_action, history, pr)
        if (total_steps % STORE_STEPS) == 0:
            self.save_loc_data(total_steps)

    def save_loc_data(self, total_steps):
        old_dir = os.getcwd()
        it = 0
        try:
            os.mkdir(Path(FILEPATH, "data", "models", f"run{it}", "notSSP",f"{total_steps}","data"))
        except FileExistsError:
            print("File Exists")
        os.chdir(Path(FILEPATH, "data", "models", f"run{it}", "notSSP",f"{total_steps}","data"))
        self.save_match_history(total_steps)
        os.chdir(old_dir)
        print("Saved Match History")

    def save_match_history(self, total_steps):
        np.save('match_history', self.match_snapshot)
        self.match_snapshot = {}