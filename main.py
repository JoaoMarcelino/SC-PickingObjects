from ant_model import AntModel, average_stick_pile
import random
import os
from mesa.batchrunner import batch_run
import pandas as pd


def main():

    n_ants = [1] + [a for a in range(10, 110, 10)] 
    n_sticks = [a for a in range(50, 550, 50)] 
    neigh = ["VN","M", "IVN"]
    stick_min = [a for a in range(1,11)]
    stick_max = [a for a in range(0,11)]
    width = 10
    height = 10
    num_gens = 2000
    seed = 30

    path = "./SC-PickingObjects/results/"

    params = {
        "num_ants": n_ants,
        "num_sticks":n_sticks,
        "neighType":neigh,
        "stick_min":stick_min,
        "stick_max":stick_max,
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

    total = f"batch_1.csv"
    
    results_df.to_csv(path + total)
                        
                        
                        




if __name__ == "__main__":
    main()