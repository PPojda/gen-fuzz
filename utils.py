import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

RANKING_CAPACITY = 4
OLD_POPULATION = 0
FIRST_INPUT_RANGE = (0, 101)
SECOND_INPUT_RANGE = (0, 101)
OUTPUT_RANGE = (0, 101)
POPULATION = 10
FUZZY_TYPE = 3
MEMBERSHIP_COUNT = 3
SEED_VALUE = 45

RULES_NAMES_MAP = {4: ("low", "medium_low", "medium_high", "high"), 3: ("low", "medium", "high")}
COLUMN_RANGE_MAP = {'first_input': FIRST_INPUT_RANGE, 'second_input': SECOND_INPUT_RANGE, 'output': OUTPUT_RANGE}

input_1 = ctrl.Antecedent(np.arange(*FIRST_INPUT_RANGE, 1), 'first_input')
input_2 = ctrl.Antecedent(np.arange(*SECOND_INPUT_RANGE, 1), 'second_input')
output = ctrl.Consequent(np.arange(*OUTPUT_RANGE, 1), 'output')
output.defuzzify_method = 'centroid'


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


def create_control_system():
    rules = create_rules()
    return ctrl.ControlSystem(rules)


def simulate():
    output_ctrl = create_control_system()
    return ctrl.ControlSystemSimulation(output_ctrl)


def update_membership_functions(chromosome: pd.Series):
    global input_1, input_2, output
    for i in range(MEMBERSHIP_COUNT):
        if FUZZY_TYPE == 3:
            input_1[RULES_NAMES_MAP[MEMBERSHIP_COUNT][i]] = fuzz.trimf(input_1.universe,
                                                                       sorted(chromosome['first_input'][i]))
            input_2[RULES_NAMES_MAP[MEMBERSHIP_COUNT][i]] = fuzz.trimf(input_2.universe,
                                                                       sorted(chromosome['second_input'][i]))
            output[RULES_NAMES_MAP[MEMBERSHIP_COUNT][i]] = fuzz.trimf(output.universe, sorted(chromosome['output'][i]))
        elif FUZZY_TYPE == 4:
            input_1[RULES_NAMES_MAP[MEMBERSHIP_COUNT][i]] = fuzz.trapmf(input_1.universe,
                                                                        sorted(chromosome['first_input'][i]))
            input_2[RULES_NAMES_MAP[MEMBERSHIP_COUNT][i]] = fuzz.trapmf(input_2.universe,
                                                                        sorted(chromosome['second_input'][i]))
            output[RULES_NAMES_MAP[MEMBERSHIP_COUNT][i]] = fuzz.trapmf(output.universe, sorted(chromosome['output'][i]))
        else:
            raise ValueError("FUZZY_TYPE must be 3 or 4")


def draw_best(chromosome: pd.Series):
    update_membership_functions(chromosome)
    simulate()
    input_1.view()
    input_2.view()
    output.view()
