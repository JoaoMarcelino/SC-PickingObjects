from ant_model import AntModel
import random


def main():

    n_ants = [1, 10, 100]
    n_sticks = [50, 500, 1000]
    neigh = ["VN","M"]
    stick_min = [1, 2, 4]
    stick_max = [None, 5, 10]
    width = 10
    height = 10
    num_gens = 2000
    seed_num = 30

    path = "./SC-PickingObjects/results/"
    for seed in range(seed_num):
        for ants in n_ants:
            for sticks in n_sticks:
                for neig in neigh:
                    for min in stick_min:
                        for max in stick_max:

                            random.seed(seed)

                            model = AntModel(ants, sticks, neig, min, max, width, height)


                            for i in range(num_gens):
                                model.step()
                            
                            average_df = model.datacollector.get_model_vars_dataframe()
                            sticks_df = model.datacollector.get_agent_vars_dataframe()
                            

                            average = f"average/{ants}_{sticks}_{neig}_{min}_{max}_{num_gens}_{seed}"
                            stick = f"sticks/{ants}_{sticks}_{neig}_{min}_{max}_{num_gens}_{seed}"
                            
                            average_df.to_csv(path + average)
                            sticks_df.to_csv(path + stick)




if __name__ == "__main__":
    main()