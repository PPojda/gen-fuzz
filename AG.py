import copy
import random
from multiprocessing import Pool

import pandas as pd

import numpy as np
import skfuzzy as fuzz
from pandas import Series
from skfuzzy import control as ctrl
from sklearn.metrics import mean_squared_error

random.seed = 42
RANKING_CAPACITY = 50
OLD_POPULATION = 10
FIRST_INPUT_RANGE = (0, 11)
SECOND_INPUT_RANGE = (0, 11)
OUTPUT_RANGE = (0, 11)
POPULATION = 100
FUZZY_TYPE = 3

COLUMN_RANGE_MAP = {'first_input': FIRST_INPUT_RANGE, 'second_input': SECOND_INPUT_RANGE, 'output': OUTPUT_RANGE}

input_1 = ctrl.Antecedent(np.arange(*FIRST_INPUT_RANGE, 1), 'first_input')
input_2 = ctrl.Antecedent(np.arange(*SECOND_INPUT_RANGE, 1), 'second_input')
output = ctrl.Consequent(np.arange(*OUTPUT_RANGE, 1), 'output')
output.defuzzify_method = 'som'


def input_csv(input):
    with open(input) as f:
        return f.readlines()


def build_starting_chromosome():
    first_variable = [[random.randint(*FIRST_INPUT_RANGE) for _ in range(FUZZY_TYPE)],
                      [random.randint(*FIRST_INPUT_RANGE) for _ in range(FUZZY_TYPE)]]
    second_variable = [[random.randint(*SECOND_INPUT_RANGE) for _ in range(FUZZY_TYPE)],
                       [random.randint(*SECOND_INPUT_RANGE) for _ in range(FUZZY_TYPE)]]
    output_variable = [[random.randint(*OUTPUT_RANGE) for _ in range(FUZZY_TYPE)],
                       [random.randint(*OUTPUT_RANGE) for _ in range(FUZZY_TYPE)],
                       [random.randint(*OUTPUT_RANGE) for _ in range(FUZZY_TYPE)]]

    chromosome = [*first_variable, *second_variable, *output_variable]
    return chromosome


def mutate(chromosome: pd.Series):
    mutated_chromosome = copy.deepcopy(chromosome)
    for column, order in mutated_chromosome.index:
        r = random.uniform(0, 1)
        if r > 0.5:
            i = random.randint(0, FUZZY_TYPE - 1)
            mutated_chromosome.loc[column, order][i] = random.randint(*COLUMN_RANGE_MAP[column])
    return mutated_chromosome


def rate_chromosome(chromosome: pd.Series, input_values, outputs, output_sim):
    update_membership_functions(chromosome)
    return rating_function(input_values, outputs, output_sim)


def build_starting_population(input_values, outputs):
    columns = [('first_input', 1), ('first_input', 2),
               ('second_input', 1), ('second_input', 2),
               ('output', 1), ('output', 2), ('output', 3)]
    columns = pd.MultiIndex.from_tuples(columns, names=('type', 'order'))

    starting_chromosomes = [build_starting_chromosome() for _ in range(POPULATION)]
    starting_chromosomes = pd.DataFrame(
        columns=columns,
        data=starting_chromosomes)

    return starting_chromosomes


def crossing(first_chromosome: Series, second_chromosome: Series):
    new_chromosome = first_chromosome.copy()
    for column, order in first_chromosome.index:
        if order % 2 == random.randint(0, 1):
            new_chromosome[column, order] = copy.deepcopy(second_chromosome[column, order])
        else:
            new_chromosome[column, order] = copy.deepcopy(first_chromosome[column, order])
    return new_chromosome


def cross_and_mutate(parents):
    first_chromosome, second_chromosome = parents
    child = crossing(first_chromosome, second_chromosome)
    child = mutate(child)
    return child


def generate_new_population(parents_population):
    parents_population.drop('ratings', axis=1, inplace=True, level='type')
    children_population = pd.DataFrame(columns=parents_population.columns, index=range(POPULATION - OLD_POPULATION))

    with Pool() as pool:
        parent_pairs = ((parents_population.iloc[random.randint(0, RANKING_CAPACITY - 1)],
                         parents_population.iloc[random.randint(0, RANKING_CAPACITY - 1)])
                        for _ in range(len(children_population)))

        children_population = pool.map(cross_and_mutate, parent_pairs)
    new = pd.concat([pd.DataFrame(children_population), parents_population.iloc[:OLD_POPULATION]], ignore_index=True)
    return new


def define_initial_membership_functions():
    input_1['low'] = fuzz.trimf(input_1.universe, [0, 0, 0])
    input_1['high'] = fuzz.trimf(input_1.universe, [0, 0, 0])

    input_2['low'] = fuzz.trimf(input_2.universe, [0, 0, 0])
    input_2['high'] = fuzz.trimf(input_2.universe, [0, 0, 0])

    output['low'] = fuzz.trimf(output.universe, [0, 0, 0])
    output['medium'] = fuzz.trimf(output.universe, [0, 0, 0])
    output['high'] = fuzz.trimf(output.universe, [0, 0, 0])


def create_rules():
    rule1 = ctrl.Rule(input_1['high'] & input_2['high'], output['high'])
    rule2 = ctrl.Rule(input_1['high'] & input_2['low'], output['medium'])
    rule3 = ctrl.Rule(input_1['low'] & input_2['high'], output['medium'])
    rule4 = ctrl.Rule(input_1['low'] & input_2['low'], output['low'])

    return [rule1, rule2, rule3, rule4]


def create_control_system():
    rules = create_rules()
    output_ctrl = ctrl.ControlSystem(rules)
    return output_ctrl


def initialize_simulation():
    define_initial_membership_functions()
    output_ctrl = create_control_system()
    return ctrl.ControlSystemSimulation(output_ctrl)


def update_membership_functions(chromosome: pd.Series):
    input_1['low'] = fuzz.trimf(input_1.universe, sorted(chromosome['first_input'][1]))
    input_1['high'] = fuzz.trimf(input_1.universe, sorted(chromosome['first_input'][2]))
    input_2['low'] = fuzz.trimf(input_2.universe, sorted(chromosome['second_input'][1]))
    input_2['high'] = fuzz.trimf(input_2.universe, sorted(chromosome['second_input'][2]))

    output['low'] = fuzz.trimf(output.universe, sorted(chromosome['output'][1]))
    output['medium'] = fuzz.trimf(output.universe, sorted(chromosome['output'][2]))
    output['high'] = fuzz.trimf(output.universe, sorted(chromosome['output'][3]))


def rating_function(inputs, outputs, output_sim):
    fuzz_outputs = []

    for first, second in zip(*inputs):
        output_sim.input['first_input'] = first
        output_sim.input['second_input'] = second
        output_sim.compute()
        fuzz_outputs.append(output_sim.output['output'])

    return float(mean_squared_error(fuzz_outputs, outputs))


if __name__ == '__main__':
    input_data = input_csv('output.csv')
    first_values = [int(row.split(',')[0]) for row in input_data]
    second_values = [int(row.split(',')[1]) for row in input_data]
    output_values = [float(row.split(',')[2].strip()) for row in input_data]

    simulation = initialize_simulation()
    rating_function((first_values, second_values), outputs=output_values, output_sim=simulation)

    population = build_starting_population(input_values=(first_values, second_values), outputs=output_values)
    population['ratings'] = population.apply(
            lambda row: rate_chromosome(row, (first_values, second_values), outputs=output_values, output_sim=simulation), axis=1
        )
    population.sort_values(by=['ratings'], inplace=True)

    count = 0
    while count < 10:
        population.drop(population.iloc[RANKING_CAPACITY:].index, inplace=True)
        new_population = generate_new_population(population.copy())
        with Pool() as pool:
            new_population['ratings'] = pool.starmap(rate_chromosome, [(row, (first_values, second_values), output_values, simulation)
                                                                       for _, row in new_population.iterrows()])
        new_population.sort_values(by=['ratings'], inplace=True)
        if population.iloc[0]['ratings'].values[0] <= new_population.iloc[0]['ratings'].values[0]:
            count += 1
        else:
            count = 0
        population = new_population
        print(population.iloc[0])
