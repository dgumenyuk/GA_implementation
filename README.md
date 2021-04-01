# Single objective algorithm implementation in python
 The goal is to generate test cases for a cyber-physical system (smart thermostat).
 The initial test cases are included in the "final_json" file. The genetic algorithm optimizes the test cases, so that they reveal the "weak" points of the system.
 The goal function is the distance between the expected vs simulated system behaviour. To calculate the fitness function each test case is executed using the developped smart thermostat model.
 The algorithm can be launched with the command: `"python3 ga_class.py"`
