import logging
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule

from ant_model import AntModel


def agent_portrayal(agent):

    if agent.name == "Ant":
        portrayal = {"Shape": "circle",
                    "Filled": "true",
                    "r": 0.5,
                    "Layer": 0}
        if agent.stick == None:
            portrayal["Color"] = "red"
        else:
            portrayal["Color"] = "grey"

    elif agent.name == "Stick":
        portrayal = {"Shape": "circle",
            "Filled": "true",
            "r": 0.2,
            "Layer": 1,
            "Color": "brown"}

    return portrayal



if __name__ == "__main__":
    
    width = 10
    height = 10

    # Moore, Van Neumann, Inverse Van Neumann
    neigh = ["M", "VN", "IVN"]

    grid = CanvasGrid(agent_portrayal, width, height, 500, 500)

    #chart = ChartModule([{"Label": "Gini",
    #                    "Color": "Black"}],
    #                    data_collector_name='datacollector')

    server = ModularServer(AntModel,
                        [grid],
                        "Ant Model",
                        {"num_ants":10, "num_sticks":40, "neighType":neigh[0], "width":width, "height":height})  
                        
    server.port = 8521 # The default
    server.launch()