from enum import Enum
import random

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


random.seed = 43
MEMBERSHIP_COUNT = 3


class DefuzzificationMethod(Enum):
    centroid = 'centroid'
    bisector = 'bisector'
    mean_of_maximum = 'mom'
    min_of_maximum = 'som'
    max_of_maximum = 'lom'


# Step 1: Define the fuzzy sets for input variables (cost and benefit)
input_1 = ctrl.Antecedent(np.arange(0, 101, 1), 'first_input')
input_2 = ctrl.Antecedent(np.arange(0, 101, 1), 'second_input')
output = ctrl.Consequent(np.arange(0, 101, 1), 'output')
output.defuzzify_method = DefuzzificationMethod.centroid.value


def create_membership_functions():
    global input_1, input_2, output
    if MEMBERSHIP_COUNT == 4:
        input_1['low'] = fuzz.trimf(input_1.universe, [0, 0, 30])
        input_1['medium_low'] = fuzz.trimf(input_1.universe, [20, 40, 6])
        input_1['medium_high'] = fuzz.trimf(input_1.universe, [4, 6, 8])
        input_1['high'] = fuzz.trimf(input_1.universe, [7, 10, 10])
        input_2['low'] = fuzz.trimf(input_2.universe, [0, 0, 3])
        input_2['medium_low'] = fuzz.trimf(input_2.universe, [2, 4, 6])
        input_2['medium_high'] = fuzz.trimf(input_2.universe, [4, 6, 8])
        input_2['high'] = fuzz.trimf(input_2.universe, [7, 10, 10])
        output['low'] = fuzz.trimf(output.universe, [0, 0, 3])
        output['medium_low'] = fuzz.trimf(output.universe, [2, 4, 6])
        output['medium_high'] = fuzz.trimf(output.universe, [4, 6, 8])
        output['high'] = fuzz.trimf(output.universe, [7, 10, 10])
    else:
        input_1['low'] = fuzz.trimf(input_1.universe, [0, 20, 50])
        input_1['medium'] = fuzz.trimf(input_1.universe, [10, 50, 90])
        input_1['high'] = fuzz.trimf(input_1.universe, [50, 80, 100])
        input_2['low'] = fuzz.trimf(input_2.universe, [0, 20, 50])
        input_2['medium'] = fuzz.trimf(input_2.universe, [10, 50, 90])
        input_2['high'] = fuzz.trimf(input_2.universe, [50, 80, 100])
        output['low'] = fuzz.trimf(output.universe, [0, 20, 50])
        output['medium'] = fuzz.trimf(output.universe, [10, 50, 90])
        output['high'] = fuzz.trimf(output.universe, [50, 80, 100])


create_membership_functions()

# Step 3: Define the fuzzy rules
def create_rules():
    if MEMBERSHIP_COUNT == 3:
        rule1 = ctrl.Rule(input_1['high'] & input_2['high'], output['low'])
        rule2 = ctrl.Rule(input_1['high'] & input_2['medium'], output['low'])
        rule3 = ctrl.Rule(input_1['high'] & input_2['low'], output['medium'])
        rule4 = ctrl.Rule(input_1['medium'] & input_2['high'], output['medium'])
        rule5 = ctrl.Rule(input_1['medium'] & input_2['medium'], output['medium'])
        rule6 = ctrl.Rule(input_1['medium'] & input_2['low'], output['high'])
        rule7 = ctrl.Rule(input_1['low'] & input_2['high'], output['high'])
        rule8 = ctrl.Rule(input_1['low'] & input_2['medium'], output['high'])
        rule9 = ctrl.Rule(input_1['low'] & input_2['low'], output['high'])
        fallback_rule = ctrl.Rule(~(input_1['low'] | input_1['medium'] | input_1['high']) |
                                  ~(input_2['low'] | input_2['medium'] | input_2['high']), output['medium'])
        rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, fallback_rule]
    elif MEMBERSHIP_COUNT == 4:

        rule1 = ctrl.Rule(input_1['high'] & input_2['high'], output['low'])
        rule2 = ctrl.Rule(input_1['high'] & input_2['medium_high'], output['low'])
        rule3 = ctrl.Rule(input_1['high'] & input_2['medium_low'], output['low'])
        rule4 = ctrl.Rule(input_1['high'] & input_2['low'], output['medium_low'])
        rule5 = ctrl.Rule(input_1['medium_low'] & input_2['high'], output['medium_low'])
        rule6 = ctrl.Rule(input_1['medium_low'] & input_2['medium_high'], output['medium_low'])
        rule7 = ctrl.Rule(input_1['medium_low'] & input_2['medium_low'], output['medium_high'])
        rule8 = ctrl.Rule(input_1['medium_low'] & input_2['low'], output['medium_high'])
        rule9 = ctrl.Rule(input_1['medium_high'] & input_2['high'], output['medium_high'])
        rule10 = ctrl.Rule(input_1['medium_high'] & input_2['medium_high'], output['medium_high'])
        rule11 = ctrl.Rule(input_1['medium_high'] & input_2['medium_low'], output['high'])
        rule12 = ctrl.Rule(input_1['medium_high'] & input_2['low'], output['high'])
        rule13 = ctrl.Rule(input_1['low'] & input_2['high'], output['high'])
        rule14 = ctrl.Rule(input_1['low'] & input_2['medium_high'], output['high'])
        rule15 = ctrl.Rule(input_1['low'] & input_2['medium_low'], output['high'])
        rule16 = ctrl.Rule(input_1['low'] & input_2['low'], output['high'])

        fallback_rule = ctrl.Rule(~(input_1['low'] | input_1['high'] | input_1['medium_low'] | input_1['medium_high']) |
                                  ~(input_2['low'] | input_2['high'] | input_2['medium_low'] | input_2['medium_high']),
                                  output['medium_low'])
        rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14,
                 rule15, rule16, fallback_rule]
    else:
        raise ValueError("MEMBERSHIP_COUNT must be 3 or 4")
    return rules


# Step 4: Implement the fuzzy inference system
output_ctrl = ctrl.ControlSystem(create_rules())
output_sim = ctrl.ControlSystemSimulation(output_ctrl)

output_sim.input['first_input'] = 5
output_sim.input['second_input'] = 5
output_sim.compute()

# print("Output: ", output_sim.output['output'])
input_1.view()
input_2.view()
output.view()


def generate_input():
    first = np.random.randint(0, 101)
    second = np.random.randint(0, 101)
    return first, second


def evaluate_output(input_1, input_2):
    output_sim.input['first_input'] = input_1
    output_sim.input['second_input'] = input_2
    output_sim.compute()
    return output_sim.output['output']


# Generate random inputs and evaluate the output
outputs = set()
for i in range(1000):
    input_1, input_2 = generate_input()
    output_value = evaluate_output(input_1, input_2)
    outputs.add(output_value)
    with open('output.csv', 'a') as f:
        f.write("{},{},{}\n".format(input_1, input_2, output_value))
print(len(outputs))
