from asyncio.windows_events import NULL
from asyncore import read
from turtle import width
from ant_model import AntModel, average_stick_pile
import random
import os
from mesa.batchrunner import batch_run
import pandas as pd
from os.path import exists
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import json

def main():
    

    n_ants = [1] + [a for a in range(10, 60, 10)] 
    n_sticks = [a for a in range(50, 300, 50)] 
    neigh = ["VN","M", "IVN"]
    stick_min = [1, 2]
    stick_max = [5, 10, 15]
    width = 10
    height = 10
    num_gens = 1000
    seed = 30

    path = "./results/"

    for ants in n_ants:
        for sticks in n_sticks:
            for neighbor in neigh:
                for s_min in stick_min:
                    for s_max in stick_max:

                        
                        total = f"batch_{ants}_{sticks}_{neighbor}_{s_min}_{s_max}.csv"
                        path_to_file = path + total
                        

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
                                        

def main2():
    n_ants = 50
    n_sticks = 300 
    neigh = 'M'
    stick_min = 1
    stick_max = 0
    width = 10
    height = 10
    num_gens = 1000
    seed = 30

    colors_array = [("Blue", "Green") , ("Blue", "Green", "Yellow")]
    prob = [(0.5, 0.5), (0.34, 0.33, 0.33)]
    path = "./results/colors/"

    total = f"batch_{1}.csv"
    path_to_file = path + total
    params = {
        "num_ants": [n_ants],
        "num_sticks":[n_sticks],
        "neighType":[neigh],
        "stick_min":[stick_min],
        "stick_max":[stick_max],
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

    for i, colors in enumerate(colors_array):

        probability = prob[i]
        total = f"batch_{len(colors)}.csv"
        path_to_file = path + total
        

        print(colors, probability)
        params = {
            "num_ants": [n_ants],
            "num_sticks":[n_sticks],
            "neighType":[neigh],
            "stick_min":[stick_min],
            "stick_max":[stick_max],
            "stick_colors":[colors],
            "stick_colors_prob": [probability],
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

    path = "./results/probability/"
    probability = [(0.5, 0.5), (0.67, 0.33), (0.75, 0.25), (0.8, 0.2)]
    colors = ('blue', 'green')
    for i, prob in enumerate(probability):
        total = f"batch_{i}.csv"
        path_to_file = path + total
        

        params = {
            "num_ants": [n_ants],
            "num_sticks":[n_sticks],
            "neighType":[neigh],
            "stick_min":[stick_min],
            "stick_max":[stick_max],
            "stick_colors":[colors],
            "stick_colors_prob": [prob],
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

    path = "./results/"

    min = stick_min[0]
    max = stick_max[0]
    n = neigh[0]


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

    path = "results/"

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

    path = "./results/"

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

    path = "./results/"

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

    path = "./results/"

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

    path = "./results/"

    heatmap_info = []

    for i, ants in enumerate(n_ants):

        heatmap_line = []

        for j, sticks in enumerate(n_sticks):

            total = f"batch_{ants}_{sticks}_{n}_{min}_{max}.csv"

            df = readfile(path + total)

            average = df[(df.Step == num_gens) & (df.AgentID == 0) & (df.iteration == 0)].Average.values
        
            for j in range(1, seed):

                val = df[(df.Step == num_gens) & (df.AgentID == 0) & (df.iteration == j)].Average.values


                average = np.sum([average, val])

            average = average/seed

            heatmap_line.append(average)
        #print(heatmap_line)
        heatmap_info.append(heatmap_line)

    heatmap_info = np.array(heatmap_info)

    #print(heatmap_info)

    fig, ax = plt.subplots()
    ax = sns.heatmap(heatmap_info,annot = True,fmt = "0.3f",cmap="coolwarm")
    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(len(n_ants)), labels=n_ants)
    ax.set_xticks(np.arange(len(n_sticks)), labels=n_sticks)
    ax.set_xlabel("Sticks")
    ax.set_ylabel("Ants")
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
    #ants = 20
    #sticks= 200

    num_gens = 1000
    seed = 30

    path = "./results/"
    heatmap_info = []

    for i, ants in enumerate(n_ants):

        heatmap_line = []

        for j, sticks in enumerate(n_sticks):

            total = f"batch_{ants}_{sticks}_{n}_{min}_{max}.csv"

            df = readfile(path + total)

            average = df[(df.Step == num_gens) & (df.AgentID == 0) & (df.iteration == 0)].Median.values
        
            for j in range(1, seed):

                val = df[(df.Step == num_gens) & (df.AgentID == 0) & (df.iteration == j)].Median.values

                #print(val, average)

                average = np.sum([average, val])

            average = (average/seed)
            heatmap_line.append(average)
        #print(heatmap_line)
        heatmap_info.append(heatmap_line)

    heatmap_info = np.array(heatmap_info)
    #print(heatmap_info)

    fig, ax = plt.subplots()
    ax = sns.heatmap(heatmap_info,annot = True,fmt = "0.3f",cmap="YlGnBu")

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(len(n_ants)), labels=n_ants)
    ax.set_xticks(np.arange(len(n_sticks)), labels=n_sticks)
    ax.set_xlabel("Sticks")
    ax.set_ylabel("Ants")
    ax.set_title("Final Median of Sticks for x sticks and y ants")
    fig.tight_layout()
    plt.show()

def heatmap_Average_MinMax():
    
    n_ants = [1] + [a for a in range(10, 60, 10)] 
    n_sticks = [a for a in range(50, 300, 50)]
    neigh = ["VN","M", "IVN"]
    stick_min = [1, 2, 5]
    stick_max = [0, 5, 10, 15]
    
    #min = stick_min[0]
    #max = stick_max[0]
    n = neigh[0]
    ants = 50
    sticks= 250

    num_gens = 1000
    seed = 30

    path = "./results/"

    heatmap_info = []

    for i, min in enumerate(stick_min):

        heatmap_line = []

        for j, max in enumerate(stick_max):
            if (max == 5 and min == 5):
                heatmap_line.append(np.NaN)
                continue

            total = f"batch_{ants}_{sticks}_{n}_{min}_{max}.csv"

            df = readfile(path + total)

            average = df[(df.Step == num_gens) & (df.AgentID == 0) & (df.iteration == 0)].Average.values
        
            for j in range(1, seed):

                val = df[(df.Step == num_gens) & (df.AgentID == 0) & (df.iteration == j)].Average.values


                average = np.sum([average, val])

            average = average/seed

            heatmap_line.append(average)
        #print(heatmap_line)
        heatmap_info.append(heatmap_line)

    heatmap_info = np.array(heatmap_info)

    #print(heatmap_info)

    fig, ax = plt.subplots()
    ax = sns.heatmap(heatmap_info,annot = True,fmt = "0.3f",cmap="coolwarm")
    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(len(stick_min)), labels=stick_min)
    ax.set_xticks(np.arange(len(stick_max)), labels=stick_max)
    ax.set_xlabel("stick max of stack")
    ax.set_ylabel("Stick min to form stack")
    ax.set_title("Final Average of Sticks for x max of stack and y min to form stack")
    fig.tight_layout()
    plt.show()

def heatmap_Median_MinMax():
    
    n_ants = [1] + [a for a in range(10, 60, 10)] 
    n_sticks = [a for a in range(50, 300, 50)]
    neigh = ["VN","M", "IVN"]
    stick_min = [1, 2, 5]
    stick_max = [0, 5, 10, 15]
    
    min = stick_min[0]
    max = stick_max[0]
    n = neigh[0]
    ants = 50
    sticks= 250

    num_gens = 1000
    seed = 30

    path = "./results/"

    heatmap_info = []

    for i, min in enumerate(stick_min):

        heatmap_line = []

        for j, max in enumerate(stick_max):

            if (max == 5 and min == 5):
                heatmap_line.append(np.NaN)
                continue
            total = f"batch_{ants}_{sticks}_{n}_{min}_{max}.csv"

            df = readfile(path + total)

            average = df[(df.Step == num_gens) & (df.AgentID == 0) & (df.iteration == 0)].Median.values
        
            for j in range(1, seed):

                val = df[(df.Step == num_gens) & (df.AgentID == 0) & (df.iteration == j)].Median.values

                #print(val, average)

                average = np.sum([average, val])

            average = (average/seed)
            heatmap_line.append(average)
        #print(heatmap_line)
        heatmap_info.append(heatmap_line)

    heatmap_info = np.array(heatmap_info)
    #print(heatmap_info)

    fig, ax = plt.subplots()
    ax = sns.heatmap(heatmap_info,annot = True,fmt = "0.3f",cmap="YlGnBu")

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(len(stick_min)), labels=stick_min)
    ax.set_xticks(np.arange(len(stick_max)), labels=stick_max)
    ax.set_xlabel("stick max of stack")
    ax.set_ylabel("Stick min to form stack")
    ax.set_title("Final Median of Sticks for x max of stack and y min to form stack")
    fig.tight_layout()
    plt.show()

def plotMultipleSticks():

    n_ants = 50
    n_sticks = 300
    neigh = "M"
    stick_min = 1
    stick_max = 0
    width = 10
    height = 10
    num_gens = 1000
    seed = 30

    colors_array = [["Blue"], ["Blue", "Green"] , ["Blue", "Green", "Yellow"]]
    prob = [(0.5, 0.5), (0.34, 0.33, 0.33)]
    path = "./results/colors/"


    fig, axs = plt.subplots(1, 1, sharey=True, sharex=True)
    fig.suptitle(f"Alteração de StickMax para {ants} ants e {sticks} sticks")

    plt.xlabel("Number of Generations")
    plt.ylabel("Average of Sticks")

    for i, colors in enumerate(colors_array):


        total = f"batch_{i+ 1}.csv"
        path_to_file = path + total

        df = readfile(path + total)

        step0 = (df.Step== 0) & (df.iteration == 0)
        agentId = (df.AgentID == 0) & (df.iteration == 0)

        steps =  df[(step0) | (agentId)].Step.values
        averageTotal = df[(step0) | (agentId)].AverageTotal.values

        averages = df[(step0) | (agentId)].AverageByColor.values
        averages = strToInt(averages)


        for j in range(1, seed):

            step0 = (df.Step== 0) & (df.iteration == j)
            agentId = (df.AgentID == 0) & (df.iteration == j)

            df_averageTotal =  df[step0 | agentId].AverageTotal.values

            df_averages = df[step0 | agentId].AverageByColor.values
            df_averages = strToInt(df_averages)


            averageTotal = np.sum([averageTotal, df_averageTotal], axis=0)
            averages = np.sum([averages, df_averages], axis = 0)
        #print(averages)
        averageTotal = averageTotal/seed
        averages = averages/seed

        axs.plot(steps, averageTotal,  label='Total', color='Black')
        print(averages.shape)
        for i in range(averages.shape[1]):
            print(steps.shape, np.transpose(averages)[i].shape)
            axs.plot(steps,  np.transpose(averages)[i],  label=f'{colors[i]}', color = colors[i])
        
        axs.legend(loc="upper right")


    plt.show()


    path = "./results/probability/"
    probability = [(0.5, 0.5), (0.67, 0.33), (0.75, 0.25), (0.8, 0.2)]



def strToInt(array):
    final = []
    for elem in array:
        av = json.loads(elem)
        final.append(av)
    return np.array(final)

if __name__ == "__main__":
    #main()

    n_sticks = [a for a in range(50, 300, 50)] 
    for sticks in n_sticks:
        #plotAnts(sticks)
        pass
    
    n_ants = [1] + [a for a in range(10, 60, 10)] 
    for ants in n_ants:
        #plotSticks(ants)
        pass


    ant = 50
    sticks = 250

    #plotNeigh(1, sticks)
    #plotNeigh(50, sticks)

    #plotStickMin(1, sticks)
    #plotStickMin(ant, sticks)

    #plotStickMax(1, sticks)
    #plotStickMax(30, 200)
    #plotStickMax(20, 250)
    #plotStickMax(50, 200)
    #plotStickMax(50, 250)
    
    
    #heatmap_Average()
    #heatmap_Median()

    #heatmap_Average_MinMax()
    #heatmap_Median_MinMax()

    #main2()
    plotMultipleSticks()
