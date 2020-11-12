import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import json

class Thermostat:
    """Class for performing spam filtering; using vocabular.json file as
    well as CRUD and EmalAnalyze classes"""

    def __init__(self, sample_rate=1, cmd_lst_prob=0, sns_noise_prob=0):
        self.start_temp = 20
        self.current_step = 1
        self.current_temp = 20
        self.total_temp_list = []
        self.total_step_list = []
        self.total_ideal_temp_list = []
        self.total_ideal_step_list = []
        self.total_step = 1
        self.comfort_time = 0
        self.smpl_rate = sample_rate
        self.cmd_lst_prob = cmd_lst_prob
        self.sns_noise_prob = sns_noise_prob

    def build_image(self, tc_id):
        fig, ax1 = plt.subplots(figsize=(24, 8))
        #ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        #ax1.plot(np.array(time_list), np.array(tries1), linestyle='None', marker='o')
        #rms_ = self.calculate_rms()
        ax1.plot([i*3/60 for i in self.total_step_list], self.total_temp_list, 'or', label='Actual temperature')
        #ax1.plot([i*3/60 for i in self.total_step_list], self.create_corresponding_list(1), 'or', label='Actual temperature')
        corsp_list = self.create_corresponding_list()
        discomf = self.get_discomfort(self.total_temp_list, corsp_list)
        rmse = self.calculate_rms(self.total_temp_list, corsp_list)
        ax1.set_title('Temperature values, modeled vs expected, discomf_% = '+str(discomf*100) + ', rmse = ' + str(rmse)
         + ', cmd_lst_prob = '+str(self.cmd_lst_prob) + ', sns_noise_prob = '+str(self.sns_noise_prob)
         + ', smpl_rate = '+str(self.smpl_rate), fontsize=17)
        ax1.set_xlabel('Time, hours', fontsize=14)
        ax1.set_ylabel('Temperature value in degrees Celsius', fontsize=14)
        top = 28
        bottom = 15
        ax1.set_ylim(bottom, top)
        
        #ax1.set_xticks(time_list)
        #ax1.set_xticklabels(time_list, rotation=45, fontsize=12)
        plt.yticks(np.arange(bottom, top+1, 1.0), fontsize=12)
        ax1.plot([i*3/60 for i in self.total_ideal_step_list], self.total_ideal_temp_list, '--b', label='Scheduled temperature')
        #ax1.plot([i*3/60 for i in self.total_step_list], self.create_corresponding_list(), 'ob', label='Scheduled temperature')
            #fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
        plt.grid(b=True, which='major', axis='both')

        ax1.legend(fontsize=14)
        ctime = int(time.time()) 
            
        fig.savefig('./new_model_smp_rate'+str(self.smpl_rate)+str(tc_id)+'_' + str(ctime) + '.png')
        #fig.savefig('./new_model_graphs/'+str(tc_id)+'_' + str(ctime) + '.png')
        plt.close(fig)



    def write_to_csv(self):
        with open("temp_file.csv", mode="w") as file:
            writer = csv.writer(file, delimiter=",")

            writer.writerow(
                ["Temperature, °C","Ideal temperature, °C", "Time, steps", "Time, min", "Time, sec"]
            )
            i = 0
            for temp, ideal_temp, step in zip(self.total_temp_list, self.total_ideal_temp_list, self.total_ideal_step_list):
                writer.writerow([temp, ideal_temp, step, step * 3, step * 3 * 60])
                i += 1
        return True

    def add_values_to_list(self, input_list, to_list):
        for value in input_list:
            to_list.append(value)


    def temp_inc(self, start_temp, current_step):
        transitions = ["norm", "anomal"]
        tran_matrix = [1-self.sns_noise_prob, self.sns_noise_prob]
        anomal_temp_list = [40, 100]
        action = np.random.choice(transitions, p=tran_matrix)

        #result_temp = (
        #    2.8633585562082224
        #    + 2.20569908 * math.log(current_step)
        #    + 0.8152207 * start_temp
        #temp = 10*(1 - np.exp(-0.07408519 * (current_step))) + start_temp
        temp = 7*(1 - np.exp(-0.13480706 * current_step)) + start_temp 
        #temp = 6*(1 - np.exp(-0.17733787 * current_step)) + start_temp
        if action == "norm":
            virt_temp = 7*(1 - np.exp(-0.13480706 * current_step)) + start_temp 
            #virt_temp = 10*(1 - np.exp(-0.07408519 * (current_step))) + start_temp
        else:
            virt_temp = np.random.choice(anomal_temp_list) 
       # print("ON ", virt_temp)
        #return temp, virt_temp
        return temp, virt_temp

    def temp_dec(self, start_temp, current_step):
        transitions = ["norm", "anomal"]
        tran_matrix = [1-self.sns_noise_prob, self.sns_noise_prob]
        anomal_temp_list = [-5, 0]
        action = np.random.choice(transitions, p=tran_matrix)

        # Y = 6.174526226174903 - 1.31384781*ln(X1) + 0.77467416*X2
        #result_temp = (
        #    6.174526226174903
        #    - 1.31384781 * math.log(current_step)
        #    + 0.77467416 * start_temp
        #)
        #temp = 5.6*(np.exp(-0.02929884*current_step)) + start_temp - 5.6
        temp = 6*(np.exp(-0.02929884*current_step)) + start_temp - 6 
        if action == "norm":
            virt_temp = 6*(np.exp(-0.02929884*current_step)) + start_temp - 6
        else:
            virt_temp = np.random.choice(anomal_temp_list) 

       # print("OFF ", virt_temp)
        #return temp, virt_temp
        return temp, virt_temp

    def set_temp(self, goal_temp, steps):
        #temp_list = []
        ideal_temp_list = []

     #   start_temp = self.start_temp
       # print("current_temp ", self.current_temp)
      #  ideal_temp_list.append(start_temp)
        #temp_list.append(start_temp)
       # self.total_ideal_step_list.append(self.total_step)

        while self.current_step < steps:
            ideal_temp_list.append(goal_temp)
            self.current_temp = goal_temp
            #temp_list.append(self.current_temp)
            self.current_step += 1
            self.total_step += 1
            self.total_ideal_step_list.append(self.total_step)
        self.current_step = 1
        #self.add_values_to_list(temp_list, self.total_temp_list)
        self.add_values_to_list(ideal_temp_list, self.total_ideal_temp_list)
        self.start_temp = self.current_temp
        #print("Temperature set to ", self.current_temp)
        return self.current_temp



    def calculate_rms(self, list1, list2):
        #list1 = np.array(self.total_temp_list)
        #list2 = np.array(self.total_ideal_temp_list)
        if len(list1) > len(list2):
            list2.append(list2[len(list2)-1])
            print("L1>L2")
        if len(list1) < len(list2):
            list1.append(list1[len(list1)-1])
            print("L1<L2")
        return np.sqrt(((np.array(list1) - np.array(list2)) ** 2).mean())

    def get_discomfort(self, actual_tmp, corsp_ideal_tmp):
        if len(actual_tmp) > len(corsp_ideal_tmp):
            corsp_ideal_tmp.append(corsp_ideal_tmp[len(corsp_ideal_tmp)-1])
            print("L1>L2")
        if len(actual_tmp) < len(corsp_ideal_tmp):
            actual_tmp.append(actual_tmp[len(actual_tmp)-1])
            print("L1<L2")
        i = 0
        margin = 1.5
        if len(actual_tmp) != len(corsp_ideal_tmp):
            print("Lists of different size")
            return False
        discomfort_time = 0
        while i < len(actual_tmp):
            if (actual_tmp[i] > corsp_ideal_tmp[i] + margin) or (actual_tmp[i] < corsp_ideal_tmp[i] - margin):
                discomfort_time += 1
            i += 1
        discomfort_time_percent = discomfort_time/len(actual_tmp)
        return discomfort_time_percent



    def create_corresponding_list(self):
        new_list = []
        new_list.append(self.total_ideal_temp_list[0])
        i = 1
        while i < len(self.total_ideal_temp_list)-self.smpl_rate:
            new_list.append(self.total_ideal_temp_list[i])
            i += self.smpl_rate
        return new_list

    def execute_test_case(self, test_case):
        for state in test_case:
            self.set_temp(test_case[state]["temp"], test_case[state]["duration"])

        #self.build_image(tc_id)

    def write_statistics(self, file, tc_id, actual_tmp, correct_tmp):
        discomfort = self.get_discomfort(actual_tmp, correct_tmp)*100
        rms = self.calculate_rms(actual_tmp, correct_tmp)
        with open(file, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerow([tc_id, discomfort, rms])

        return





    def generate_system_temp(self, tc_id):
        #system_temp_start = self.total_ideal_temp_list[0]
        #system_temp = 18 #self.total_ideal_temp_list[0]
        #system_temp = 16
        system_temp = self.total_ideal_temp_list[0]
        self.total_temp_list.append(system_temp)
        #self.total_temp_list.append(18)
        i = 1
        print(len(self.total_ideal_temp_list)*3/60)
        transitions = ["inc", "dec"]
        #inc_tran_matrix = [0.95, 0.05]
        #dec_tran_matrix = [0.05, 0.95]
        inc_tran_matrix = [1-self.cmd_lst_prob, self.cmd_lst_prob]
        dec_tran_matrix = [self.cmd_lst_prob, 1-self.cmd_lst_prob]

       # smpl_rate = 1
        self.total_step_list.append(1)
        virt_temp = system_temp
        while i < len(self.total_ideal_temp_list) - self.smpl_rate:
            #step = 1
            start_temp = system_temp
            #virt_temp = start_temp
            if(virt_temp < self.total_ideal_temp_list[i]+0.5): #+0.5
                action = np.random.choice(transitions, p=inc_tran_matrix)
                if action == 'inc':
                    system_temp, i, virt_temp = self.increase_process(virt_temp, start_temp, i, 0)
                elif action == 'dec':
                    system_temp, i, virt_temp = self.decrease_process(virt_temp, start_temp, i,  1)
                else:
                    print("error")
            #step = 1
            start_temp = system_temp

            if(virt_temp >= self.total_ideal_temp_list[i]-0.5): #-0.5
                action = np.random.choice(transitions, p=dec_tran_matrix)
                if action == 'inc':
                    system_temp, i, virt_temp = self.increase_process(virt_temp, start_temp, i,  1)
                elif action == 'dec':
                    system_temp, i, virt_temp = self.decrease_process(virt_temp, start_temp, i,  0)
                else:
                    print("error")


        #self.build_image(tc_id)


    def increase_process(self, system_temp, start_temp, i,  pass_):
        step = self.smpl_rate
        virt_temp = system_temp
        while ((virt_temp < self.total_ideal_temp_list[i]+0.5) and (i < len(self.total_ideal_temp_list)-self.smpl_rate) or ((pass_) )):
            system_temp, virt_temp = self.temp_inc(start_temp, step)
         #   print("Goal", self.total_ideal_temp_list[i])
            self.total_temp_list.append(system_temp)
            i += self.smpl_rate
            self.total_step_list.append(i)
            #print("Inc", i)
            step += self.smpl_rate
            pass_ = 0
            if(i >= len(self.total_ideal_temp_list)):
                return system_temp, i, virt_temp


        return system_temp, i, virt_temp

    def decrease_process(self, system_temp, start_temp, i, pass_):
        step = self.smpl_rate
        #if(pass_):
            #print("pass")
        virt_temp = system_temp
        while ((virt_temp >= self.total_ideal_temp_list[i]-0.5) and (i < len(self.total_ideal_temp_list)-self.smpl_rate) or ((pass_))):
            system_temp, virt_temp = self.temp_dec(start_temp, step)
        #    print("Goal", self.total_ideal_temp_list[i])
            self.total_temp_list.append(system_temp)
            i += self.smpl_rate
            self.total_step_list.append(i)
            #print("Dec", i)
            step += self.smpl_rate
            pass_ = 0

        return system_temp, i, virt_temp
        



def choose_parent(test_cases):
    candidates = list(np.random.choice(100, 4))
    fitness_function_result = {}

    for num in candidates:
        test_case = test_cases["tc"+str(num)]
        tc_name = "tc"+str(num)
        score = fitness_function(test_case)
        fitness_function_result[tc_name] = score

    maximum = 0
    for tc in fitness_function_result:
        if fitness_function_result[tc] > maximum:
            maximum = fitness_function_result[tc]
            maximum_tc = tc
        print("case", fitness_function_result[tc])

    print(maximum)
    return test_cases[maximum_tc]



def fitness_function(test_case):
    therm = Thermostat()
    therm.execute_test_case(test_case)
    therm.generate_system_temp(test_case)
    corsp_list = therm.create_corresponding_list()
    #discomf = self.get_discomfort(self.total_temp_list, corsp_list)
    rmse = therm.calculate_rms(therm.total_temp_list, corsp_list)
    return rmse

def do_crossover(parent1, parent2):
    options = ["def", "cross"]
    options_matrix = [0.2, 0.8]
    action = np.random.choice(options, p=options_matrix)
    if action == "def":
        return parent1, parent2
    elif action == "cross":
        child1 = {}
        child2 = {}
        for i in range(0, 3, 1):
            child1["st"+str(i)] = parent1["st"+str(i)]
            child2["st"+str(i)] = parent2["st"+str(i)]
        for i in range(3, len(parent2), 1):
            child1["st"+str(i)] = parent2["st"+str(i)]
        for i in range(3, len(parent1), 1):
            child2["st"+str(i)] = parent1["st"+str(i)]
        return child1, child2
    else:
        return False

def do_mutation(child):
    options = ["def", "mut1", "mut2"]
    options_matrix = [0.8, 0.1, 0.1]
    action = np.random.choice(options, p=options_matrix)
    if action == "def":
        return child
    elif action == "mut1":
        candidates = list(np.random.randint(1, high=len(child), size=2))
        temp = child['st'+str(candidates[0])]
        child['st'+str(candidates[0])] = child['st'+str(candidates[1])]
        child['st'+str(candidates[1])] = temp
        return child
    elif action == "mut2":
        num = int(np.random.randint(1, high=len(child), size=1))
        value = np.random.choice(['duration', 'temp'])
        if value == 'duration':
            child['st'+str(num)][value] = np.random.choice([100, 100, 100, 60, 60, 60, 200, 150, 150, 55, 55, 220])
        if value == 'temp':
            child['st'+str(num)][value] = np.random.choice([16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
        return child
    else:
        print("error")
        return False





if __name__ == "__main__":

    with open("final_test_cases.json") as file:
        test_cases = json.load(file)
    fitness_population_size = 20
    parent_dict = {}
    children_dict = {}
    for i in range(0, fitness_population_size, 1):
        parent = choose_parent(test_cases)
        parent_dict[str(i)] = parent
    #print(parent_dict)
        child1, child2 = do_crossover(parent_dict[str(0)], parent_dict[str(1)])
    child1 = do_mutation(child1)
    child2 = do_mutation(child2)




 
    print("children")
    print(child1)
    print(child2)
    print("mutation")
    child1 = do_mutation(child1)
    print(child1)




