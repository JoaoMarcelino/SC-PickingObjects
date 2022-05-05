from asyncio.windows_events import NULL
from asyncore import read
from ant_model import AntModel, average_stick_pile
import random
import os
from mesa.batchrunner import batch_run
import pandas as pd
from os.path import exists
import matplotlib.pyplot as plt

import numpy as np


def main():
    

    n_ants = [1] + [a for a in range(10, 60, 10)] 
    n_sticks = [a for a in range(50, 300, 50)] 
    neigh = ["VN","M", "IVN"]
    stick_min = [5]
    stick_max = [10, 15, 0]
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
                                        
                        

def readfile(file):
    return pd.read_csv(file)

def plotAnts(sticks):

    n_ants = [1] + [a for a in range(10, 60, 10)] 
    n_sticks = [a for a in range(50, 300, 50)] 
    neigh = ["VN","M", "IVN"]
    stick_min = [1, 2, 5]
    stick_max = [0, 5, 10, 15]
    width = 10
    height = 10
    num_gens = 1000
    seed = 30

    path = "./SC-PickingObjects/results/"

    min = stick_min[0]
    max = stick_max[0]
    n = neigh[0]


    size_ants = len(n_ants)
    size_sticks = len(n_sticks)
    fig, axs = plt.subplots(1, 1, sharey=True, sharex=True)
    fig.suptitle(f"Alteração de ants para {sticks} sticks")

    plt.xlabel("Number of Generations")
    plt.ylabel("Average of Sticks")

    for i, ants in enumerate(n_ants):
        total = f"batch_{ants}_{sticks}_{n}_{min}_{max}.csv"
        df = readfile(path + total)

        step0 = (df.Step== 0) & (df.iteration == 0)
        agentId = (df.AgentID == 0) & (df.iteration == 0)

        steps =  df[(step0) | (agentId)].Step.values
        average = df[(step0) | (agentId)].Average.values
        
        for j in range(1, seed):

            step0 = (df.Step== 0) & (df.iteration == j)
            agentId = (df.AgentID == 0) & (df.iteration == j)

            df_average =  df[step0 | agentId].Average.values

            average = np.sum([average, df_average], axis=0)

        average = average/seed

        axs.plot(steps, average,  label=f'{ants}')
        axs.legend(loc="upper right")


    plt.show()

def plotSticks(ants):

    n_ants = [1] + [a for a in range(10, 60, 10)] 
    n_sticks = [a for a in range(50, 300, 50)] 
    neigh = ["VN","M", "IVN"]
    stick_min = [1, 2, 5]
    stick_max = [0, 5, 10, 15]
    width = 10
    height = 10
    num_gens = 1000
    seed = 30

    path = "./SC-PickingObjects/results/"

    min = stick_min[0]
    max = stick_max[0]
    n = neigh[0]


    size_ants = len(n_ants)
    size_sticks = len(n_sticks)
    fig, axs = plt.subplots(1, 1, sharey=True, sharex=True)
    fig.suptitle(f"Alteração de sticks para {ants} ants")

    plt.xlabel("Number of Generations")
    plt.ylabel("Average of Sticks")

    for i, sticks in enumerate(n_sticks):
        total = f"batch_{ants}_{sticks}_{n}_{min}_{max}.csv"
        df = readfile(path + total)

        step0 = (df.Step== 0) & (df.iteration == 0)
        agentId = (df.AgentID == 0) & (df.iteration == 0)

        steps =  df[(step0) | (agentId)].Step.values
        average = df[(step0) | (agentId)].Average.values
        
        for j in range(1, seed):

            step0 = (df.Step== 0) & (df.iteration == j)
            agentId = (df.AgentID == 0) & (df.iteration == j)

            df_average =  df[step0 | agentId].Average.values

            average = np.sum([average, df_average], axis=0)

        average = average/seed

        axs.plot(steps, average,  label=f'{sticks}')
        axs.legend(loc="upper right")


    plt.show()



def plotNeigh(ants, sticks):

    n_ants = [1] + [a for a in range(10, 60, 10)] 
    n_sticks = [a for a in range(50, 300, 50)] 
    neigh = ["VN","M", "IVN"]
    stick_min = [1, 2, 5]
    stick_max = [0, 5, 10, 15]
    width = 10
    height = 10
    num_gens = 1000
    seed = 30

    path = "./SC-PickingObjects/results/"

    min = stick_min[0]
    max = stick_max[0]



    size_ants = len(n_ants)
    size_sticks = len(n_sticks)
    fig, axs = plt.subplots(1, 1, sharey=True, sharex=True)
    fig.suptitle(f"Alteração de Neighborhood para {ants} ants e {sticks} sticks")

    plt.xlabel("Number of Generations")
    plt.ylabel("Average of Sticks")

    for i, n in enumerate(neigh):
        total = f"batch_{ants}_{sticks}_{n}_{min}_{max}.csv"
        df = readfile(path + total)

        step0 = (df.Step== 0) & (df.iteration == 0)
        agentId = (df.AgentID == 0) & (df.iteration == 0)

        steps =  df[(step0) | (agentId)].Step.values
        average = df[(step0) | (agentId)].Average.values
        
        for j in range(1, seed):

            step0 = (df.Step== 0) & (df.iteration == j)
            agentId = (df.AgentID == 0) & (df.iteration == j)

            df_average =  df[step0 | agentId].Average.values

            average = np.sum([average, df_average], axis=0)

        average = average/seed

        axs.plot(steps, average,  label=f'{n}')
        axs.legend(loc="upper right")


    plt.show()

def plotStickMin(ants, sticks):

    n_ants = [1] + [a for a in range(10, 60, 10)] 
    n_sticks = [a for a in range(50, 300, 50)] 
    neigh = ["VN","M", "IVN"]
    stick_min = [1, 2, 5]
    stick_max = [0, 5, 10, 15]
    width = 10
    height = 10
    num_gens = 1000
    seed = 30

    path = "./SC-PickingObjects/results/"

    n = neigh[0]
    max = stick_max[0]



    size_ants = len(n_ants)
    size_sticks = len(n_sticks)
    fig, axs = plt.subplots(1, 1, sharey=True, sharex=True)
    fig.suptitle(f"Alteração de StickMin para {ants} ants e {sticks} sticks")

    plt.xlabel("Number of Generations")
    plt.ylabel("Average of Sticks")

    for i, min in enumerate(stick_min):
        total = f"batch_{ants}_{sticks}_{n}_{min}_{max}.csv"
        df = readfile(path + total)

        step0 = (df.Step== 0) & (df.iteration == 0)
        agentId = (df.AgentID == 0) & (df.iteration == 0)

        steps =  df[(step0) | (agentId)].Step.values
        average = df[(step0) | (agentId)].Average.values
        
        for j in range(1, seed):

            step0 = (df.Step== 0) & (df.iteration == j)
            agentId = (df.AgentID == 0) & (df.iteration == j)

            df_average =  df[step0 | agentId].Average.values

            average = np.sum([average, df_average], axis=0)

        average = average/seed

        axs.plot(steps, average,  label=f'{min}')
        axs.legend(loc="upper right")


    plt.show()

def plotStickMax(ants, sticks):

    n_ants = [1] + [a for a in range(10, 60, 10)] 
    n_sticks = [a for a in range(50, 300, 50)] 
    neigh = ["VN","M", "IVN"]
    stick_min = [1, 2, 5]
    stick_max = [0, 5, 10, 15]
    width = 10
    height = 10
    num_gens = 1000
    seed = 30

    path = "./SC-PickingObjects/results/"

    n = neigh[0]
    min = stick_min[0]

    size_ants = len(n_ants)
    size_sticks = len(n_sticks)
    fig, axs = plt.subplots(1, 1, sharey=True, sharex=True)
    fig.suptitle(f"Alteração de StickMax para {ants} ants e {sticks} sticks")

    plt.xlabel("Number of Generations")
    plt.ylabel("Average of Sticks")

    for i, max in enumerate(stick_max):
        total = f"batch_{ants}_{sticks}_{n}_{min}_{max}.csv"
        df = readfile(path + total)

        step0 = (df.Step== 0) & (df.iteration == 0)
        agentId = (df.AgentID == 0) & (df.iteration == 0)

        steps =  df[(step0) | (agentId)].Step.values
        average = df[(step0) | (agentId)].Average.values
        
        for j in range(1, seed):

            step0 = (df.Step== 0) & (df.iteration == j)
            agentId = (df.AgentID == 0) & (df.iteration == j)

            df_average =  df[step0 | agentId].Average.values

            average = np.sum([average, df_average], axis=0)

        average = average/seed

        axs.plot(steps, average,  label=f'{max}')
        axs.legend(loc="upper right")


    plt.show()


def heatmap_Average():
    
    n_ants = [1] + [a for a in range(10, 60, 10)] 
    n_sticks = [a for a in range(50, 300, 50)]
    neigh = ["VN","M", "IVN"]
    stick_min = [1, 2, 5]
    stick_max = [0, 5, 10, 15]
    
    min = stick_min[0]
    max = stick_max[0]
    n = neigh[0]

    num_gens = 1000
    seed = 30

    path = "./SC-PickingObjects/results/"

    heatmap_info = []

    for i, ants in enumerate(n_ants):

        heatmap_line = []

        for j, sticks in enumerate(n_sticks):

            total = f"batch_{ants}_{sticks}_{n}_{min}_{max}.csv"

            df = readfile(path + total)

            average = df[(df.Step == num_gens) & (df.AgentID == 0) & (df.iteration == 0)].Average.values
        
            for j in range(1, seed):

                val = df[(df.Step == num_gens) & (df.AgentID == 0) & (df.iteration == j)].Average.values


                average = np.sum([average, val], axis=0)

            average = average/seed

            heatmap_line.append(average)
        #print(heatmap_line)
        heatmap_info.append(heatmap_line)

    heatmap_info = np.array(heatmap_info)

    #print(heatmap_info)

    fig, ax = plt.subplots()
    im = ax.imshow(heatmap_info)

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(len(n_ants)), labels=n_ants)
    ax.set_xticks(np.arange(len(n_sticks)), labels=n_sticks)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(n_ants)):
        for j in range(len(n_sticks)):
            text = ax.text(j, i, heatmap_info[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Final Average of Sticks for x sticks and y ants")
    fig.tight_layout()
    plt.show()

def heatmap_Median():
    
    n_ants = [1] + [a for a in range(10, 60, 10)] 
    n_sticks = [a for a in range(50, 300, 50)]
    neigh = ["VN","M", "IVN"]
    stick_min = [1, 2, 5]
    stick_max = [0, 5, 10, 15]
    
    min = stick_min[0]
    max = stick_max[0]
    n = neigh[0]

    num_gens = 1000
    seed = 30

    path = "./SC-PickingObjects/results/"

    heatmap_info = []

    for i, ants in enumerate(n_ants):

        heatmap_line = []

        for j, sticks in enumerate(n_sticks):

            total = f"batch_{ants}_{sticks}_{n}_{min}_{max}.csv"

            df = readfile(path + total)

            average = df[(df.Step == num_gens) & (df.AgentID == 0) & (df.iteration == 0)].Median.values
        
            for j in range(1, seed):

                val = df[(df.Step == num_gens) & (df.AgentID == 0) & (df.iteration == j)].Median.values

                print(val, average)

                average = np.sum([average, val], axis=0)

            average = average/seed
            

            heatmap_line.append(average)
        #print(heatmap_line)
        heatmap_info.append(heatmap_line)

    heatmap_info = np.array(heatmap_info)

    #print(heatmap_info)

    fig, ax = plt.subplots()
    im = ax.imshow(heatmap_info)

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(len(n_ants)), labels=n_ants)
    ax.set_xticks(np.arange(len(n_sticks)), labels=n_sticks)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(n_ants)):
        for j in range(len(n_sticks)):
            text = ax.text(j, i, heatmap_info[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Final Median of Sticks for x sticks and y ants")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

    n_sticks = [a for a in range(50, 300, 50)] 
    for sticks in n_sticks:
        #plotAnts(sticks)
        pass
    
    n_ants = [1] + [a for a in range(10, 60, 10)] 
    for ants in n_ants:
        #plotSticks(ants)
        pass


    ant = n_ants[1]
    sticks = n_sticks[-1]

    plotNeigh(ant, sticks)
    plotStickMin(ant, sticks)
    plotStickMax(ant, sticks)
    
    
    #heatmap_Average()
    #heatmap_Median()