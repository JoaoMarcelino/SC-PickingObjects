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

def average_stick_pile(model):
    stick_piles = [agent.num_neighbors for agent in model.schedule.agents]
    N = model.num_sticks
    return sum(stick_piles) / N


class StickAgent(Agent):
    """An still agent that only moves when picked up."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.name = "Stick"
        self.ant = None
        self.num_neighbors = 0

    #CHANGES 
    def step(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        sticks = [agent for agent in cellmates if agent.name == "Stick"]

        self.num_neighbors = len(sticks)


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

        if len(free_sticks) >= self.stick_min and len(free_sticks) <= self.stick_max:
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

    def __init__(self, num_ants, num_sticks, neighType, stick_min, stick_max,  width, height):

        if stick_max == 0:
            stick_max = math.inf

        self.num_ants = num_ants
        self.num_sticks = num_sticks
        self.neighType = neighType
        self.stick_min = stick_min
        self.stick_max = stick_max

        self.grid = MultiGrid(width, height, True)

        self.schedule_ants = RandomActivation(self)
        self.schedule = RandomActivation(self)

        self.running = True

        # Create ants
        for i in range(self.num_ants):
            ant = AntAgent(i, self, self.stick_min, stick_max)
            self.schedule_ants.add(ant)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(ant, (x, y))

            self.datacollector = DataCollector(
                model_reporters={"Average": average_stick_pile}, agent_reporters={"Sticks": 'num_neighbors'}
            )
        
        # Create sticks
        for i in range(self.num_sticks):
            stick = StickAgent(i, self)

            #CHANGES
            self.schedule.add(stick)
            
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)

            self.grid.place_agent(stick, (x, y))

    def step(self):
        """Advance the model by one step."""
        self.schedule_ants.step()

        #CHANGES 
        self.schedule.step()

        self.datacollector.collect(self)