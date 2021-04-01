import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import json
from real_model_therm_stohastic import Thermostat
import random
class GA:
    """Class for performing spam filtering; using vocabular.json file as
    well as CRUD and EmalAnalyze classes"""

    def __init__(self, test_cases):


        self.child_av_list = []
        self.child_max_list = []
        self.population_av_list = []
        self.population_max_list = []
        self.iterations = []

        self.tournament_size = 4
        self.population_size = len(test_cases)
        self.crossover_rate = 0.9
        self.crossover_point = 3
        self.mutation_rate = 0.5
        self.sample_rate = 1
        self.fitness_population_size = 20
        self.elite_size = 10#int(self.population_size/4)

        self.__test_cases = test_cases
        self.__fitness_values = {}
        self.__children_dict = {}
        self.__children_fitness_dict = {}
        self.__best_test_cases = {}
        self.test_cases_hard_copy = test_cases

        self.child1 = {}
        self.child2 = {}


    
    def get_best_cases(self):
        return self.__best_test_cases


    def get_children_dict(self):
        return self.__children_dict

    def get_children_fitness_dict(self):
        return self.__children_fitness_dict

    def get_children_fitness_value(self, key):
        return self.__children_fitness_dict[key]

    def get_test_case(self, key):
        result = self.__test_cases[key]
        return result

    def get_all_test_cases(self):
        result = self.__test_cases
        return result

    def get_all_fitness_values(self):
        return self.__fitness_values

    def get_fitness_value(self, key):
        return self.__fitness_values[key]

    def reset_test_cases(self):
        with open("save_tc.json") as file:
            self.__test_cases = json.load(file)
        #self.__test_cases = self.test_cases_hard_copy

    def reset_best_cases(self):
        self.__best_test_cases = {}

    def modify_test_cases(self, key, value):
        try:
            print("Modifying test case", key)
            self.__test_cases[key] = value
        except:
            return False

        return True

    def set_fitness_values(self, key, value):
        try:
            self.__fitness_values[key] = value
        except:
            return False

        return True

    def set_children_dict(self, key, value):
        try:
            self.__children_dict[key] = value
        except:
            return False

        return True

    def zero_children_dict(self):
        try:
            self.__children_dict = {}
        except:
            return False

        return True

    def set_children_fitness_dict(self, key, value):
        try:
            self.__children_fitness_dict[key] = value
        except:
            return False

        return True


    def zero_children_fitness_dict(self):
        try:
            self.__children_fitness_dict = {}
        except:
            return False

        return True

    def set_best_cases(self, key, value):
        try:
            self.__best_test_cases[key] = value
        except:
            return False

        return True



    def choose_parent(self):
        candidates = list(np.random.choice(self.population_size, self.tournament_size))
        fitness_function_result = {}

        for num in candidates:
            tc_name = "tc"+str(num)
            score = self.get_fitness_value("fitness_tc"+str(num))  # fitness_function(test_case)
            fitness_function_result[tc_name] = score

        maximum = 0
        for tc in fitness_function_result:
            if fitness_function_result[tc] > maximum:
                maximum = fitness_function_result[tc]
                maximum_tc = tc

        result = self.get_all_test_cases()
        return result[maximum_tc]

    def build_image(self, test_case, tc_id, folder):
        therm = Thermostat(self.sample_rate, 0, 0) 
        therm.execute_test_case(test_case)
        therm.generate_system_temp()
        #corsp_list = therm.create_corresponding_list()
        therm.build_image(tc_id, folder)
        return

    def fitness_function(self, test_case, img=0, tc_id=0):
        therm = Thermostat(self.sample_rate, 0, 0) 
        therm.execute_test_case(test_case)
        therm.generate_system_temp()
        corsp_list = therm.create_corresponding_list()
        if(img):
            therm.build_image(tc_id)
           # print("****fitness_func")
           # print("test case in fnction", test_case)
           # print("***_fitness end")
        rmse = therm.calculate_rms(therm.total_temp_list, corsp_list)
        return rmse

    def do_crossover(self, parent1, parent2):
        options = ["def", "cross"]
        options_matrix = [1-self.crossover_rate, self.crossover_rate]
        action = np.random.choice(options, p=options_matrix)
        if action == "def":
            return parent1, parent2
        elif action == "cross":
            child1 = {}
            child2 = {}
            for i in range(0, self.crossover_point, 1):
                child1["st"+str(i)] = parent1["st"+str(i)]
                child2["st"+str(i)] = parent2["st"+str(i)]
            for i in range(self.crossover_point, len(parent2), 1):
                child1["st"+str(i)] = parent2["st"+str(i)]
            for i in range(self.crossover_point, len(parent1), 1):
                child2["st"+str(i)] = parent1["st"+str(i)]
            return child1, child2
        else:
            return False

    def do_mutation(self, child):
        #print("AT MUTATION START", self.get_all_test_cases())
        self.test_cases_hard_copy = self.get_all_test_cases()
        #print("HARD COPY",  self.test_cases_hard_copy)

        with open("save_tc.json", "w+") as outfile:
            json.dump(self.test_cases_hard_copy, outfile)

        options = ["def", "mut1", "mut2"]
        options_matrix = [1 - self.mutation_rate, self.mutation_rate/2, self.mutation_rate/2]
        action = np.random.choice(options, p=options_matrix)
        #action = "mut2"
        #action = "mut2"
        if action == "def":
            #print("mutation DEF")
            return child
        elif action == "mut1":

            #print("mutation MUT1")
            candidates = list(np.random.randint(1, high=len(child), size=2))
            temp = child['st'+str(candidates[0])]
            child['st'+str(candidates[0])] = child['st'+str(candidates[1])]
            child['st'+str(candidates[1])] = temp
            self.reset_test_cases()
            return child

        elif action == "mut2":
            #print("mutation MUT2")
            num = int(np.random.randint(1, high=len(child), size=1))
            value = np.random.choice(['duration', 'temp'])

            if value == 'duration':
                child['st'+str(num)][value] = int(np.random.choice([100, 100, 100, 60, 60, 60, 200, 150, 150, 55, 55, 220]))
            if value == 'temp':
                maximum = 25
                minimum = 16
                #value = np.random.choice(['duration', 'temp'])
                action = np.random.choice(['inc', 'dec'])
                old_value = child['st'+str(num-1)][value]
                #print("******Old value", old_value)
               # print("******action", action)
                if action == "inc":
                    
                    new_value = random.randint(old_value, maximum)
                    #print("******new_value", new_value)
                    child['st'+str(num)][value] = new_value
                elif action == "dec":
                    diff = old_value - minimum
                    if diff > 5:
                        new_value = random.randint(old_value-6, old_value)
                    else:
                        new_value = random.randint(minimum, old_value)
                   # print("******new_value", new_value)
                    child['st'+str(num)][value] = new_value
                    #new_value = random.randint(old_value, maximum)
                #new_value = int(np.random.choice([16, 17, 18, 19, 20, 21, 22, 23, 24, 25]))
                #if new_value < old_value:
                  #  new_value = int(np.random.choice([old_value-1, old_value-2, old_value-3, old_value-4, old_value-5]))
                    #child['st'+str(num)][value] = new_value
                #else:
                    #child['st'+str(num)][value] = int(np.random.choice([16, 17, 18, 19, 20, 21, 22, 23, 24, 25]))

            self.reset_test_cases()
            return child
        else:
            print("error")
            return False



    def create_new_population(self):

        old_population = self.get_all_test_cases()
        old_fitness = self.get_all_fitness_values()
        
        children_dict = self.get_children_dict()
        children_fitness = self.get_children_fitness_dict()
        for child in children_dict:
            minimum = 100
            for tc in old_population:
                fitness = old_fitness["fitness_"+tc]
                if fitness < minimum:
                    minimum = fitness
                    minimum_tc = tc

            #print("minimum tc ", minimum_tc)
            #print("old:", self.get_fitness_value("fitness_"+minimum_tc))
            #print(self.get_test_case(minimum_tc))
            if self.get_fitness_value("fitness_"+minimum_tc) < children_fitness["fitness_"+child]:
                self.modify_test_cases(minimum_tc, children_dict[child])
                self.set_fitness_values("fitness_"+minimum_tc, children_fitness["fitness_"+child])
                print(children_fitness["fitness_"+child])

                #print("new:", self.get_fitness_value("fitness_"+minimum_tc))
               # print("new_checked", self.fitness_function(self.get_test_case(minimum_tc)))
                #print(self.get_test_case(minimum_tc))
            else:
                #print("new:", self.get_fitness_value("fitness_"+minimum_tc))
                #print(self.get_test_case(minimum_tc))
                continue

        return self.get_all_test_cases()

    def run_iteration(self):
        parent_dict = {}
        self.zero_children_dict()
        self.zero_children_fitness_dict() 

        for i in range(0, self.fitness_population_size-1, 1):
            parent_dict[str(i)] = self.choose_parent()
            parent_dict[str(i+1)] = self.choose_parent()

            child1, child2 = self.do_crossover(parent_dict[str(i)], parent_dict[str(i+1)])

            child1_ = self.do_mutation(child1)

            child2_ = self.do_mutation(child2)

            self.set_children_dict("tc"+str(i), child1_)
            self.set_children_fitness_dict("fitness_tc"+str(i), self.fitness_function(child1_))
            self.set_children_dict("tc"+str(i+1), child2_)
            self.set_children_fitness_dict("fitness_tc"+str(i+1), self.fitness_function(child2_))
            i += 2
        
        return True

    def evaluate_children(self):
        maximum = 0
        all_scores = []
        children_dict = self.get_children_dict()
        for tc in children_dict:
            fitness = self.get_children_fitness_value("fitness_"+tc)
            all_scores.append(fitness)
            if fitness > maximum:
                maximum = fitness
                maximum_tc = tc
        print("Children max fitness: ", maximum)

        print("Children average fitness: ", sum(all_scores)/len(all_scores))
        self.child_max_list.append(maximum)
        self.child_av_list.append(sum(all_scores)/len(all_scores))

        return True

    def evaluate_population(self):
        maximum = 0
        all_scores = []
        population = self.get_all_test_cases()
        for tc in population:

            fitness = self.get_fitness_value("fitness_"+tc)
            all_scores.append(fitness)
            if fitness > maximum:
                maximum = fitness

        print("Max population fitness: ", maximum)
        print("Average population fitness: ", sum(all_scores)/len(all_scores))

        self.population_max_list.append(maximum)
        self.population_av_list.append(sum(all_scores)/len(all_scores))

        return True

    def add_fitness(self):

        test_cases = self.get_all_test_cases()
        for tc in test_cases:
            test_case = self.get_test_case(tc)
            self.set_fitness_values("fitness_"+tc, self.fitness_function(test_case))

        return True

    def select_best(self, folder):
        maximum = 0
        max_list = []
        self.reset_best_cases()
        maximum_dict = self.get_best_cases()
        test_cases = self.get_all_test_cases()
        for i in range(0, self.elite_size, 1):
            #print(i)
            for tc in test_cases:
                fitness = self.get_fitness_value("fitness_"+tc) 
                maximum_dict = self.get_best_cases()
                if ((fitness > maximum) and (tc not in maximum_dict) and (fitness not in max_list)):
                    maximum = fitness
                    maximum_tc = tc
            max_list.append(maximum)
            print("***Evaluation function")
            print(maximum_tc)
            print(maximum)
            #print("in function", self.get_test_case(maximum_tc))
            self.set_best_cases(maximum_tc, self.get_test_case(maximum_tc))
            print("***evaluation end")
            maximum = 0

        best_cases = self.get_best_cases()
        for test_case in best_cases:
            self.build_image(best_cases[test_case], test_case, folder)

        return self.get_best_cases()

    def plot_progress(self):
        x = self.iterations
        y_ch_av = self.child_av_list
        y_ch_mx = self.child_max_list
        y_p_av = self.population_av_list
        y_p_mx = self.population_max_list
        fig, ax = plt.subplots(figsize=(17, 7))
        #ax.plot(x, y, linestyle='None', marker='o',label='Children average fitness function')
        #ax.plot(x, y_ch_av, linestyle='None', marker='o', label='Population average fitness function')
        #ax.plot(x, y_ch_mx, linestyle='None', marker='o', label='Population average fitness function')
        ax.plot(x, y_p_av, linestyle='None', marker='o', label='Population average fitness function')
        ax.plot(x, y_p_mx, linestyle='None', marker='o', label='Population maximum fitness function')

        ax.set_ylabel('Root mean square error (fintess function)', fontsize=14)
        ax.set_xlabel('Iteration', fontsize=14)
        #ax.set_xticks(x_pos)
        #ax.set_xticklabels(materials)
        #ax.set_title('RSSI vs distance')
        #ax.set_title('Tries vs distance')
        ax.set_title('Fitness function', fontsize=14)
        ax.yaxis.grid(True)
        plt.grid(b=True, which='major', axis='both')

        ax.legend(fontsize=14)
        # Save the figure and show
        plt.tight_layout()
        plt.savefig('./results/evolution.png')
        #plt.savefig('TriesvsDistance.png')
        #plt.savefig('TriesvsRSSI.png')
        plt.show()



if __name__ == "__main__":

    with open("final_test_cases.json") as file:
        test_cases = json.load(file)

    #print(test_cases)

    gen = GA(test_cases)
    gen.add_fitness()
    #gen.select_best()
    #print(gen.get_all_fitness_values())
    print("Initial population")
    gen.evaluate_population()
    #gen.select_best("initial")
    it_num = 30

    i = 1
    delta_min = 0.01
    count = 0 
    gen.iterations = [0]
    #for i in range(1, it_num+1, 1):
    while count < 3:
        print("+++++++++++++++++++++++++++++++++")
        print("ITERATION:", i)
        gen.run_iteration()
        gen.evaluate_children()
        gen.evaluate_population()
        gen.iterations.append(i)
        gen.create_new_population()
        
        #print("All cases")
        #print(gen.get_all_test_cases())
        print("++++++++++++++++++++++++++++++++")
        delta = gen.population_av_list[i] - gen.population_av_list[i-1]
        if delta < delta_min:
            count += 1
        i += 1
        #i += 1

    print("ch_av", gen.child_av_list)
    print("ch_mx", gen.child_max_list)
    print("pop_av", gen.population_av_list)
    print("pop_mx", gen.population_max_list)

    best_cases = gen.select_best("best")
    #for case in best_cases:
        #rmse = gen.fitness_function(best_cases[case], img=1, tc_id=case)
      #  print("main ***")
      #  print("case", case)
     #   print("rmse_new", rmse)
        #print("in main", best_cases[case])
      #  print("main end***")
    gen.plot_progress()


    #with open("best.json", "w+") as outfile:
     #   json.dump(test_cases, outfile)

