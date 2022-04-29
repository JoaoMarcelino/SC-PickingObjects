from ant_model import AntModel, average_stick_pile
import random
import os
from mesa.batchrunner import batch_run
import pandas as pd
from os.path import exists



def main():
    

    n_ants = [1] + [a for a in range(10, 60, 10)] 
    n_sticks = [a for a in range(50, 300, 50)] 
    neigh = ["VN","M", "IVN"]
    stick_min = [1, 2, 5]
    stick_max = [5, 10, 15, 0]
    width = 10
    height = 10
    num_gens = 1000
    seed = 30

    path = "./SC-PickingObjects/results/"

    for ants in n_ants:
        for sticks in n_sticks:
            for neighbor in neigh:
                for s_min in stick_min:
                    for s_max in stick_max:


                        if s_max == s_min:
                            break
                        
                        total = f"batch_{ants}_{sticks}_{neighbor}_{s_min}_{s_max}.csv"
                        path_to_file = path + total
                        
                        if exists(path_to_file):
                            break

                        params = {
                            "num_ants": [ants],
                            "num_sticks":[sticks],
                            "neighType":[neighbor],
                            "stick_min":[s_min],
                            "stick_max":[s_max],
                            "width":width,
                            "height":height
                        }
                        
                        results = batch_run(
                        AntModel,
                        parameters=params,
                        iterations=seed,
                        max_steps=num_gens,
                        number_processes=None,
                        data_collection_period=50,
                        display_progress=True,
                        )

                        results_df = pd.DataFrame(results)
                        
                        results_df.to_csv(path_to_file)
                                        
                        





if __name__ == "__main__":
    main()