import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

RANKING_CAPACITY = 300
OLD_POPULATION = 5
FIRST_INPUT_RANGE = (0, 101)
SECOND_INPUT_RANGE = (0, 101)
OUTPUT_RANGE = (0, 101)
POPULATION = 300
FUZZY_TYPE = 3
MEMBERSHIP_COUNT = 3
SEED_VALUE = 45
MUTATE_RANGE = 0.3

RULES_NAMES_MAP = {4: ("low", "medium_low", "medium_high", "high"), 3: ("low", "medium", "high")}
COLUMN_RANGE_MAP = {'first_input': FIRST_INPUT_RANGE, 'second_input': SECOND_INPUT_RANGE, 'output': OUTPUT_RANGE}

fuzzy_input_1 = ctrl.Antecedent(np.arange(*FIRST_INPUT_RANGE, 1), 'first_input')
fuzzy_input_2 = ctrl.Antecedent(np.arange(*SECOND_INPUT_RANGE, 1), 'second_input')
fuzzy_output = ctrl.Consequent(np.arange(*OUTPUT_RANGE, 1), 'output')
fuzzy_output.defuzzify_method = 'centroid'


def create_rules(fuzzy_input_1, fuzzy_input_2, fuzzy_output):
    if MEMBERSHIP_COUNT == 3:
        rule1 = ctrl.Rule(fuzzy_input_1['high'] & fuzzy_input_2['high'], fuzzy_output['low'])
        rule2 = ctrl.Rule(fuzzy_input_1['high'] & fuzzy_input_2['medium'], fuzzy_output['low'])
        rule3 = ctrl.Rule(fuzzy_input_1['high'] & fuzzy_input_2['low'], fuzzy_output['medium'])
        rule4 = ctrl.Rule(fuzzy_input_1['medium'] & fuzzy_input_2['high'], fuzzy_output['medium'])
        rule5 = ctrl.Rule(fuzzy_input_1['medium'] & fuzzy_input_2['medium'], fuzzy_output['medium'])
        rule6 = ctrl.Rule(fuzzy_input_1['medium'] & fuzzy_input_2['low'], fuzzy_output['high'])
        rule7 = ctrl.Rule(fuzzy_input_1['low'] & fuzzy_input_2['high'], fuzzy_output['high'])
        rule8 = ctrl.Rule(fuzzy_input_1['low'] & fuzzy_input_2['medium'], fuzzy_output['high'])
        rule9 = ctrl.Rule(fuzzy_input_1['low'] & fuzzy_input_2['low'], fuzzy_output['high'])
        fallback_rule = ctrl.Rule(~(fuzzy_input_1['low'] | fuzzy_input_1['medium'] | fuzzy_input_1['high']) |
                                  ~(fuzzy_input_2['low'] | fuzzy_input_2['medium'] | fuzzy_input_2['high']),
                                  fuzzy_output['medium'])
        rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, fallback_rule]
    elif MEMBERSHIP_COUNT == 4:
        rule1 = ctrl.Rule(fuzzy_input_1['high'] & fuzzy_input_2['high'], fuzzy_output['low'])
        rule2 = ctrl.Rule(fuzzy_input_1['high'] & fuzzy_input_2['medium_high'], fuzzy_output['low'])
        rule3 = ctrl.Rule(fuzzy_input_1['high'] & fuzzy_input_2['medium_low'], fuzzy_output['low'])
        rule4 = ctrl.Rule(fuzzy_input_1['high'] & fuzzy_input_2['low'], fuzzy_output['medium_low'])
        rule5 = ctrl.Rule(fuzzy_input_1['medium_low'] & fuzzy_input_2['high'], fuzzy_output['medium_low'])
        rule6 = ctrl.Rule(fuzzy_input_1['medium_low'] & fuzzy_input_2['medium_high'], fuzzy_output['medium_low'])
        rule7 = ctrl.Rule(fuzzy_input_1['medium_low'] & fuzzy_input_2['medium_low'], fuzzy_output['medium_high'])
        rule8 = ctrl.Rule(fuzzy_input_1['medium_low'] & fuzzy_input_2['low'], fuzzy_output['medium_high'])
        rule9 = ctrl.Rule(fuzzy_input_1['medium_high'] & fuzzy_input_2['high'], fuzzy_output['medium_high'])
        rule10 = ctrl.Rule(fuzzy_input_1['medium_high'] & fuzzy_input_2['medium_high'], fuzzy_output['medium_high'])
        rule11 = ctrl.Rule(fuzzy_input_1['medium_high'] & fuzzy_input_2['medium_low'], fuzzy_output['high'])
        rule12 = ctrl.Rule(fuzzy_input_1['medium_high'] & fuzzy_input_2['low'], fuzzy_output['high'])
        rule13 = ctrl.Rule(fuzzy_input_1['low'] & fuzzy_input_2['high'], fuzzy_output['high'])
        rule14 = ctrl.Rule(fuzzy_input_1['low'] & fuzzy_input_2['medium_high'], fuzzy_output['high'])
        rule15 = ctrl.Rule(fuzzy_input_1['low'] & fuzzy_input_2['medium_low'], fuzzy_output['high'])
        rule16 = ctrl.Rule(fuzzy_input_1['low'] & fuzzy_input_2['low'], fuzzy_output['high'])
        fallback_rule_1 = ctrl.Rule(~(fuzzy_input_2['low'] | fuzzy_input_2['high'] | fuzzy_input_2['medium_low'] | fuzzy_input_2['medium_high']),
                                    fuzzy_output['medium_high'])
        fallback_rule_2 = ctrl.Rule(~(fuzzy_input_1['low'] | fuzzy_input_1['high'] | fuzzy_input_1['medium_low'] | fuzzy_input_1['medium_high']),
                                    fuzzy_output['medium_low'])
        rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14,
                 rule15, rule16, fallback_rule_1, fallback_rule_2]
    else:
        raise ValueError("MEMBERSHIP_COUNT must be 3 or 4")
    return rules


def create_control_system(fuzzy_input_1, fuzzy_input_2, fuzzy_output):
    rules = create_rules(fuzzy_input_1, fuzzy_input_2, fuzzy_output)
    return ctrl.ControlSystem(rules)


def simulate(fuzzy_input_1, fuzzy_input_2, fuzzy_output):
    output_ctrl = create_control_system(fuzzy_input_1, fuzzy_input_2, fuzzy_output)
    return ctrl.ControlSystemSimulation(output_ctrl)


def update_membership_functions(chromosome: pd.Series, fuzzy_input_1, fuzzy_input_2, fuzzy_output):
    for i in range(MEMBERSHIP_COUNT):
        if FUZZY_TYPE == 3:
            fuzzy_input_1[RULES_NAMES_MAP[MEMBERSHIP_COUNT][i]] = fuzz.trimf(fuzzy_input_1.universe,
                                                                             sorted(chromosome['first_input'][i]))
            fuzzy_input_2[RULES_NAMES_MAP[MEMBERSHIP_COUNT][i]] = fuzz.trimf(fuzzy_input_2.universe,
                                                                             sorted(chromosome['second_input'][i]))
            fuzzy_output[RULES_NAMES_MAP[MEMBERSHIP_COUNT][i]] = fuzz.trimf(fuzzy_output.universe,
                                                                            sorted(chromosome['output'][i]))
        elif FUZZY_TYPE == 4:
            fuzzy_input_1[RULES_NAMES_MAP[MEMBERSHIP_COUNT][i]] = fuzz.trapmf(fuzzy_input_1.universe,
                                                                              sorted(chromosome['first_input'][i]))
            fuzzy_input_2[RULES_NAMES_MAP[MEMBERSHIP_COUNT][i]] = fuzz.trapmf(fuzzy_input_2.universe,
                                                                              sorted(chromosome['second_input'][i]))
            fuzzy_output[RULES_NAMES_MAP[MEMBERSHIP_COUNT][i]] = fuzz.trapmf(fuzzy_output.universe,
                                                                             sorted(chromosome['output'][i]))
        else:
            raise ValueError("FUZZY_TYPE must be 3 or 4")


def draw_best(chromosome: pd.Series, fuzzy_input_1, fuzzy_input_2, fuzzy_output):
    update_membership_functions(chromosome, fuzzy_input_1, fuzzy_input_2, fuzzy_output)
    simulate(fuzzy_input_1, fuzzy_input_2, fuzzy_output)
    fuzzy_input_1.view()
    fuzzy_input_2.view()
    fuzzy_output.view()
