from enum import Enum

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class DefuzzificationMethod(Enum):
    centroid = 'centroid'
    bisector = 'bisector'
    mean_of_maximum = 'mom'
    min_of_maximum = 'som'
    max_of_maximum = 'lom'


# Step 1: Define the fuzzy sets for input variables (cost and benefit)
input_1 = ctrl.Antecedent(np.arange(0, 11, 1), 'first_input')
input_2 = ctrl.Antecedent(np.arange(0, 11, 1), 'second_input')

# Membership functions for cost and benefit
input_1['low'] = fuzz.trimf(input_1.universe, [0, 0, 6])
input_1['high'] = fuzz.trimf(input_1.universe, [4, 10, 10])
input_2['low'] = fuzz.trimf(input_2.universe, [0, 0, 6])
input_2['high'] = fuzz.trimf(input_2.universe, [4, 10, 10])

# Step 2: Define the fuzzy sets for output variable (cost benefit)
output = ctrl.Consequent(np.arange(0, 11, 1), 'output')

# Membership functions for cost benefit
output['low'] = fuzz.trimf(output.universe, [0, 0, 4])
output['medium'] = fuzz.trimf(output.universe, [3, 5, 7])
output['high'] = fuzz.trimf(output.universe, [6, 10, 10])

# Step 3: Define the fuzzy rules
rule1 = ctrl.Rule(input_1['high'] & input_2['high'], output['high'])
rule2 = ctrl.Rule(input_1['high'] & input_2['low'], output['medium'])
rule3 = ctrl.Rule(input_1['low'] & input_2['high'], output['medium'])
rule4 = ctrl.Rule(input_1['low'] & input_2['low'], output['low'])

# Step 4: Implement the fuzzy inference system
output_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
output_sim = ctrl.ControlSystemSimulation(output_ctrl)

# Step 5: Test the fuzzy logic system with sample inputs
output_sim.input['first_input'] = 5
output_sim.input['second_input'] = 5

output.defuzzify_method = DefuzzificationMethod.min_of_maximum.value
output_sim.compute()

# print("Output: ", output_sim.output['output'])
input_1.view()
input_2.view()
output.view(sim=output_sim)


def generate_input():
    input_1 = np.random.randint(0, 11)
    input_2 = np.random.randint(0, 11)
    return input_1, input_2


def evaluate_output(input_1, input_2):
    output_sim.input['first_input'] = input_1
    output_sim.input['second_input'] = input_2
    output_sim.compute()
    return output_sim.output['output']


# Generate random inputs and evaluate the output
for i in range(1000):
    input_1, input_2 = generate_input()
    output_value = evaluate_output(input_1, input_2)
    with open('output.csv', 'a') as f:
        f.write("{},{},{}\n".format(input_1, input_2, output_value))
