import copy
import random
from multiprocessing import Pool

import pandas as pd

import numpy as np
import skfuzzy as fuzz
from pandas import Series
from skfuzzy import control as ctrl
from sklearn.metrics import mean_squared_error

random.seed = 45
RANKING_CAPACITY = 30
OLD_POPULATION = 5
FIRST_INPUT_RANGE = (0, 11)
SECOND_INPUT_RANGE = (0, 11)
OUTPUT_RANGE = (0, 11)
POPULATION = 100
FUZZY_TYPE = 3
MEMBERSHIP_COUNT = 3

RULES_NAMES_MAP = {2: ("low", "high"), 3: ("low", "medium", "high")}
COLUMN_RANGE_MAP = {'first_input': FIRST_INPUT_RANGE, 'second_input': SECOND_INPUT_RANGE, 'output': OUTPUT_RANGE}

input_1 = ctrl.Antecedent(np.arange(*FIRST_INPUT_RANGE, 1), 'first_input')
input_2 = ctrl.Antecedent(np.arange(*SECOND_INPUT_RANGE, 1), 'second_input')
output = ctrl.Consequent(np.arange(*OUTPUT_RANGE, 1), 'output')
output.defuzzify_method = 'mom'


def input_csv(input):
    with open(input) as f:
        return f.readlines()


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
        rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]
    elif MEMBERSHIP_COUNT == 2:
        rule1 = ctrl.Rule(input_1['high'] & input_2['high'], output['low'])
        rule2 = ctrl.Rule(input_1['high'] & input_2['low'], output['low'])
        rule3 = ctrl.Rule(input_1['low'] & input_2['high'], output['high'])
        rule4 = ctrl.Rule(input_1['low'] & input_2['low'], output['high'])
        rules = [rule1, rule2, rule3, rule4]
    else:
        raise ValueError("MEMBERSHIP_COUNT must be 2 or 3")

    return rules


def create_control_system():
    rules = create_rules()
    output_ctrl = ctrl.ControlSystem(rules)
    return output_ctrl


def simulate():
    output_ctrl = create_control_system()
    return ctrl.ControlSystemSimulation(output_ctrl)


def update_membership_functions(chromosome: pd.Series):
    global input_1, input_2, output
    for i in range(MEMBERSHIP_COUNT):
        input_1[RULES_NAMES_MAP[MEMBERSHIP_COUNT][i]] = fuzz.trimf(input_1.universe,
                                                                   sorted(chromosome['first_input'][i]))
        input_2[RULES_NAMES_MAP[MEMBERSHIP_COUNT][i]] = fuzz.trimf(input_2.universe,
                                                                   sorted(chromosome['second_input'][i]))
        output[RULES_NAMES_MAP[MEMBERSHIP_COUNT][i]] = fuzz.trimf(output.universe, sorted(chromosome['output'][i]))


def rating_function(inputs, outputs, output_sim):
    fuzz_outputs = []

    for first, second in zip(*inputs):
        output_sim.input['first_input'] = first
        output_sim.input['second_input'] = second
        output_sim.compute()
        fuzz_outputs.append(output_sim.output.get('output'))

    return mean_squared_error(outputs, fuzz_outputs)


def build_starting_chromosome():
    first_variable = []
    second_variable = []
    output_variable = []
    for i in range(MEMBERSHIP_COUNT):
        first_variable.append([random.randrange(*FIRST_INPUT_RANGE) for _ in range(FUZZY_TYPE)])
        second_variable.append([random.randrange(*SECOND_INPUT_RANGE) for _ in range(FUZZY_TYPE)])
        output_variable.append([random.randrange(*OUTPUT_RANGE) for _ in range(FUZZY_TYPE)])

    chromosome = [*first_variable, *second_variable, *output_variable]
    return chromosome


def build_starting_population(input_values, outputs):
    names = ['first_input', 'second_input', 'output']
    columns = pd.MultiIndex.from_product([names, range(0, MEMBERSHIP_COUNT)], names=('type', 'order'))

    starting_chromosomes = [build_starting_chromosome() for _ in range(POPULATION)]
    starting_chromosomes = pd.DataFrame(
        columns=columns,
        data=starting_chromosomes)

    starting_chromosomes['ratings'] = starting_chromosomes.apply(
        lambda row: rate_chromosome(row, input_values, outputs=outputs), axis=1
    )
    starting_chromosomes.sort_values(by=['ratings'], inplace=True)

    return starting_chromosomes


def mutate(chromosome: pd.Series):
    mutated_chromosome = copy.deepcopy(chromosome)
    for column, order in mutated_chromosome.index:
        r = random.uniform(0, 1)
        if r > 0.3:
            i = random.randint(0, FUZZY_TYPE - 1)
            mutated_chromosome.loc[column, order][i] = random.randrange(*COLUMN_RANGE_MAP[column])
        if order == 0 and r > 0.7:
            i = random.randint(0, FUZZY_TYPE - 1)
            j = random.randint(0, FUZZY_TYPE - 1)
            mutated_chromosome.loc[column, i], mutated_chromosome.loc[column, j] = mutated_chromosome.loc[
                column, j], mutated_chromosome.loc[column, i]
    return mutated_chromosome


def cross(first_chromosome: Series, second_chromosome: Series):
    new_chromosome = first_chromosome.copy()
    for column, order in first_chromosome.index:
        if order % 2 == random.randint(0, 1):
            new_chromosome[column, order] = copy.deepcopy(second_chromosome[column, order])
        else:
            new_chromosome[column, order] = copy.deepcopy(first_chromosome[column, order])
    return new_chromosome


def cross_alternative(first_chromosome: Series, second_chromosome: Series):
    new_chromosome = first_chromosome.copy()
    for column, order in first_chromosome.index:
        if random.randint(0, 1):
            end = random.randint(1, FUZZY_TYPE - 1)
            new_chromosome[column, order][0:end] = copy.deepcopy(second_chromosome[column, order])[0:end]
        else:
            start = random.randint(0, FUZZY_TYPE - 2)
            new_chromosome[column, order][start:2] = copy.deepcopy(first_chromosome[column, order])[start:2]
    return new_chromosome


def rate_chromosome(chromosome: pd.Series, input_values, outputs):
    update_membership_functions(chromosome)
    output_sim = simulate()
    return rating_function(input_values, outputs, output_sim)


def cross_and_mutate(parents: pd.DataFrame):
    parents.drop('ratings', axis=1, inplace=True, level='type')
    first_chromosome, second_chromosome = parents.iloc[0], parents.iloc[1]
    child = cross_alternative(first_chromosome, second_chromosome)
    child = mutate(child)
    return child


def roulette_selection(parents_population):
    inverse_ranking = 1 / parents_population['ratings']
    selection_probs = inverse_ranking / inverse_ranking.sum()
    selected_parents = parents_population.sample(n=2, weights=selection_probs, replace=False)
    return selected_parents


def generate_new_population(parents_population, pool):
    parents_pairs = [roulette_selection(parents_population) for _ in range(POPULATION - OLD_POPULATION)]
    children_population = pool.map(cross_and_mutate, parents_pairs)
    parents_population.drop('ratings', axis=1, inplace=True, level='type')
    new = pd.concat([pd.DataFrame(children_population), parents_population.iloc[:OLD_POPULATION]], ignore_index=True)
    return new


def draw_best(chromosome: pd.Series):
    update_membership_functions(chromosome)
    simulate()
    input_1.view()
    input_2.view()
    output.view()


def optimize(population: pd.DataFrame, count, limit):
    with Pool() as pool:
        while count < limit:
            best = population.iloc[0]
            population.drop(population.iloc[RANKING_CAPACITY:].index, inplace=True)
            new_population = generate_new_population(population, pool)
            new_population['ratings'] = pool.starmap(rate_chromosome,
                                                     [(row, (first_values, second_values), output_values)
                                                      for _, row in new_population.iterrows()])
            new_population.sort_values(by=['ratings'], inplace=True)
            if best['ratings'].values[0] <= new_population.iloc[0]['ratings'].values[0]:
                count += 1
            else:
                count = 0
            population = new_population
            print(population.iloc[0])
    draw_best(population.iloc[0])
    return population


if __name__ == '__main__':
    input_data = input_csv('output.csv')
    first_values = [int(row.split(',')[0]) for row in input_data]
    second_values = [int(row.split(',')[1]) for row in input_data]
    output_values = [float(row.split(',')[2].strip()) for row in input_data]

    population = build_starting_population(input_values=(first_values, second_values), outputs=output_values)

    count = 0
    limit = 15
    best_population = optimize(population, count, limit)
