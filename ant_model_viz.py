import logging
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule

from ant_model import AntModel

from collections import defaultdict

class CanvasGrid_Altered(CanvasGrid):
    def render(self, model):
        grid_state = defaultdict(list)
        for x in range(model.grid.width):
            for y in range(model.grid.height):
                cell_objects = model.grid.get_cell_list_contents([(x, y)])

                num = 0
                sticks = [agent for agent in cell_objects if agent.name == "Stick"]

                for obj in cell_objects:

                    portrayal = self.portrayal_method(obj)
                    if portrayal:
                        portrayal["x"] = x
                        portrayal["y"] = y

                        if obj.name == 'Stick':
                            num +=1
                            if num == len(sticks):
                                portrayal["layer"] = num 
                                portrayal['Number of Sticks'] = num
                                portrayal["r"] = num * 0.1

                                grid_state[portrayal["Layer"]].append(portrayal)

                        if obj.name == 'Ant':
                            grid_state[portrayal["Layer"]].append(portrayal)


        return grid_state

def agent_portrayal(agent):

    if agent.name == "Ant":
        portrayal = {"Shape": "rect",
                    "Filled": "true",
                    "w": 0.5,
                    "h":0.5,
                    "Layer": 10}
        if agent.stick == None:
            portrayal["Color"] = "red"
        else:
            portrayal["Color"] = "grey"

    elif agent.name == "Stick":
        portrayal = {"Shape": "circle",
            "Filled": "true",
            "r": 0.1,
            "Layer": 1,
            "Color": "blue"
            }

    return portrayal



if __name__ == "__main__":
    
    width = 50
    height = 50

    # Moore, Van Neumann, Inverse Van Neumann
    neigh = ["M", "VN", "IVN"]

    grid = CanvasGrid_Altered(agent_portrayal, width, height, 500, 500)

    #chart = ChartModule([{"Label": "Gini",
    #                    "Color": "Black"}],
    #                    data_collector_name='datacollector')

    server = ModularServer(AntModel,
                        [grid],
                        "Ant Model",
                        {"num_ants":10, "num_sticks":1000, "neighType":neigh[0], "stick_min": 1, "width":width, "height":height})  
                        
    server.port = 8521 # The default
    server.launch()