import logging
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.batchrunner import batch_run

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import statistics

def average_stick_pile(model):
    stick_piles = [agent.num_cellmates for agent in model.schedule.agents]
    N = model.num_sticks
    return sum(stick_piles) / N
    
def average_stick_pile_color(model):
    stick_piles = [agent for agent in model.schedule.agents]

    averages = []
    for color in model.stick_colors:
        colored_sticks = [agent.num_cellmates for agent in stick_piles if agent.color == color]
        N = len(colored_sticks)

        if (N):
            average = sum(colored_sticks)/ N
        else:
            average= 0
        averages.append(average)
    return averages

def median_stick_pile(model):
    stick_piles = [agent.num_cellmates for agent in model.schedule.agents]

    return statistics.median(stick_piles)

def median_stick_pile_color(model):
    stick_piles = [agent for agent in model.schedule.agents]

    medians = [9]
    for color in model.stick_colors:
        colored_sticks = [agent.num_cellmates for agent in stick_piles if agent.color == color]
        
        if colored_sticks:
            median = statistics.median(colored_sticks)
        else:
            median = 0.0
        medians.append(median)
    return medians


class StickAgent(Agent):
    """An still agent that only moves when picked up."""
    def __init__(self, unique_id, model, color):
        super().__init__(unique_id, model)
        self.name = "Stick"
        self.ant = None
        self.num_cellmates = 0
        self.color = color

    #CHANGES 
    def step(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        sticks = [agent for agent in cellmates if agent.name == "Stick"]

        self.num_cellmates = len(sticks)
    


class AntAgent(Agent):
    """An agent that walks randomly in the grid looking for a stick with the objective to make a pile."""

    def __init__(self, unique_id, model, stick_min, stick_max):
        super().__init__(unique_id, model)
        self.stick = None
        self.name = "Ant"
        self.stick_min = stick_min
        self.stick_max = stick_max

    def step(self):

        #Place Stick
        self.place()

        #Move Ant
        self.move()
        
        #Grab Stick
        self.grab()


    def place(self):
        """Verify if the place on the grid """

        if self.stick == None:
            return

        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        sticks = [agent for agent in cellmates if agent.name == "Stick"]
        free_sticks = [stick for stick in sticks if stick.ant == None]
        colored_sticks = [agent for agent in free_sticks if agent.color == self.stick.color]

        if len(colored_sticks) >= self.stick_min and len(colored_sticks) < self.stick_max:
            #print(f"Found Stick Pile. Dropping {self.stick.unique_id} in {self.pos}")
            self.stick.ant = None
            self.stick = None
            
    
    def move(self):
        
        #Choose Neighborhood Type
        if (self.model.neighType == "M"):
            possible_steps = self.model.grid.get_neighborhood(
                self.pos,
                moore=True,
                include_center=False)

        elif (self.model.neighType == "VN"):
            possible_steps = self.model.grid.get_neighborhood(
                self.pos,
                moore=False,
                include_center=False)

        elif (self.model.neighType == "IVN"):
            moore= self.model.grid.get_neighborhood(
                self.pos,
                moore=True,
                include_center=False)
            
            vn = self.model.grid.get_neighborhood(
                self.pos,
                moore=False,
                include_center=False)

            possible_steps = list(set(moore) - set(vn))


        #Choose Position and Move Ant and Stick
        previous_position = self.pos
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

        #logging.info(f"Ant {self.unique_id} - Moving Ant from {previous_position} to {new_position}")

        if self.stick != None:
            self.model.grid.move_agent(self.stick, new_position)

            #logging.info(f"Ant {self.unique_id} - Moving Stick from {previous_position} to  {new_position} {self.stick.pos}")
        

    def grab(self):

        if self.stick != None:
            return
        
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        sticks = [agent for agent in cellmates if agent.name == "Stick"]
        
        free_sticks = [stick for stick in sticks if stick.ant == None]
        
        if free_sticks:
            stick = self.random.choice(free_sticks)
            stick.ant = self
            self.stick = stick
            #print(f"Ant {self.unique_id} - Found Stick. Grabbing {self.stick.unique_id} in {self.pos}")

class AntModel(Model):
    """A model with 2 type of agents."""

    def __init__(self, num_ants, num_sticks, neighType, stick_min = 1, stick_max= 0, stick_colors = ['Blue'], stick_colors_prob = [1], width=10, height = 10):
        
        self.running = True

        if stick_max == 0:
            stick_max = math.inf

        if stick_colors_prob == 1 or len(stick_colors)== 1:
            stick_colors_prob = [1]

        if sum(stick_colors_prob) != 1:
            val = 1/len(stick_colors)
            stick_colors_prob = [val for color in stick_colors]
        

        self.num_ants = num_ants
        self.num_sticks = num_sticks
        self.neighType = neighType
        self.stick_min = stick_min
        self.stick_max = stick_max
        self.stick_colors = stick_colors
        self.stick_colors_prob = stick_colors_prob

        self.grid = MultiGrid(width, height, True)

        self.schedule_ants = RandomActivation(self)
        self.schedule = RandomActivation(self)



        # Create ants
        for i in range(self.num_ants):
            ant = AntAgent(i, self, self.stick_min, stick_max)
            self.schedule_ants.add(ant)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(ant, (x, y))

            self.datacollector = DataCollector(
                model_reporters={"AverageTotal": average_stick_pile, "AverageByColor": average_stick_pile_color, "MedianTotal": median_stick_pile, "MedianByColor": median_stick_pile_color,}, agent_reporters={"Sticks": 'num_cellmates'}
            )
        
        # Create sticks

        for j in range(self.num_sticks):
            prob = self.random.uniform(0, 1)
            joined_prob = 0
            for i, color_prob in enumerate(self.stick_colors_prob):
                joined_prob += color_prob

                if prob <= joined_prob:
                    color = stick_colors[i]
                    break

            stick = StickAgent(i*num_sticks + j, self, color)
            
            # Add the agent to a random grid cell
            foundStop = False
            while (not foundStop):
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)

                cellmates = self.grid.get_cell_list_contents([(x,y)])
                sticks = [agent for agent in cellmates if agent.name == "Stick"]
                colored_sticks = [agent for agent in sticks if agent.color != color]

                if not colored_sticks:
                    #print(stick.color)
                    foundStop = True

            self.grid.place_agent(stick, (x, y))
            #CHANGES
            self.schedule.add(stick)

    def step(self):
        """Advance the model by one step."""
        self.schedule_ants.step()

        #CHANGES 
        self.schedule.step()

        self.datacollector.collect(self)