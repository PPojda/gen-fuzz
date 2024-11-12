import copy
import os
import random
from multiprocessing import Pool
from sklearn.metrics import mean_squared_error
from utils import *

random.seed = SEED_VALUE


def input_csv(input):
    with open(input) as f:
        return f.readlines()


def rating_function(inputs, outputs, output_sim):
    fuzz_outputs = []

    for first, second in zip(*inputs):
        output_sim.input['first_input'] = first
        output_sim.input['second_input'] = second
        output_sim.compute()
        fuzz_outputs.append(output_sim.output.get('output'))

    return mean_squared_error(outputs, fuzz_outputs)


def correct_chromosome(chromosome: pd.Series):
    print(chromosome)
    chromosome_copy = chromosome.copy()
    for column in chromosome.index.get_level_values('type').unique():
        if column == 'ratings':
            continue
        sorted_s = chromosome_copy[column].apply(lambda x: sorted(x)[1]).sort_values().index
        chromosome_copy[column] = chromosome[column].loc[sorted_s]
    print(chromosome_copy)
    return chromosome_copy


def build_starting_chromosome():
    first_variable = []
    second_variable = []
    output_variable = []
    for i in range(MEMBERSHIP_COUNT):
        first_variable.append([random.randrange(*FIRST_INPUT_RANGE) for _ in range(FUZZY_TYPE)])
        second_variable.append([random.randrange(*SECOND_INPUT_RANGE) for _ in range(FUZZY_TYPE)])
        output_variable.append([random.randrange(*OUTPUT_RANGE) for _ in range(FUZZY_TYPE)])

    return [*first_variable, *second_variable, *output_variable]


def build_starting_population(input_values, outputs):
    names = ['first_input', 'second_input', 'output']
    columns = pd.MultiIndex.from_product([names, range(MEMBERSHIP_COUNT)], names=('type', 'order'))
    starting_chromosome = [build_starting_chromosome() for _ in range(POPULATION)]
    starting_chromosomes = pd.DataFrame(
        columns=columns,
        data=starting_chromosome)

    starting_chromosomes['ratings'] = starting_chromosomes.apply(
        lambda row: rate_chromosome(row, input_values, outputs=outputs), axis=1
    )
    starting_chromosomes.sort_values(by=['ratings'], inplace=True)

    return starting_chromosomes


def mutate(chromosome: pd.Series):
    mutated_chromosome = copy.deepcopy(chromosome)
    for column, order in mutated_chromosome.index:
        r = random.uniform(0, 1)
        if r > 0.5:
            i = random.randint(0, FUZZY_TYPE - 1)
            mutated_chromosome.loc[column, order][i] = random.randrange(*COLUMN_RANGE_MAP[column])
        if order == FUZZY_TYPE - 1 and r > 0.5:
            i = random.randint(0, MEMBERSHIP_COUNT - 1)
            j = random.randint(0, MEMBERSHIP_COUNT - 1)
            mutated_chromosome.loc[column, i], mutated_chromosome.loc[column, j] = mutated_chromosome.loc[
                column, j], mutated_chromosome.loc[column, i]
    return mutated_chromosome


def cross(first_chromosome: pd.Series, second_chromosome: pd.Series):
    new_chromosome = first_chromosome.copy()
    for column, order in first_chromosome.index:
        if order % 2 == random.randint(0, 1):
            new_chromosome[column, order] = copy.deepcopy(second_chromosome[column, order])
        else:
            new_chromosome[column, order] = copy.deepcopy(first_chromosome[column, order])
    return new_chromosome


def cross_alternative(first_chromosome: pd.Series, second_chromosome: pd.Series):
    new_chromosome = first_chromosome.copy()
    for column, order in first_chromosome.index:
        first_chromosome[column, order].sort()
        if random.randint(0, 1):
            end = random.randint(1, FUZZY_TYPE - 1)
            second_chromosome_values = copy.deepcopy(second_chromosome[column, order])
            new_chromosome[column, order][0:end] = sorted(second_chromosome_values)[0:end]
        else:
            start = random.randint(0, FUZZY_TYPE - 2)
            second_chromosome_values = copy.deepcopy(second_chromosome[column, order])
            new_chromosome[column, order][start:FUZZY_TYPE] = sorted(second_chromosome_values)[start:FUZZY_TYPE]
    return new_chromosome


def rate_chromosome(chromosome: pd.Series, input_values, outputs):
    update_membership_functions(chromosome)
    output_sim = simulate()
    return rating_function(input_values, outputs, output_sim)


def cross_and_mutate(parents: pd.DataFrame):
    parents.drop('ratings', axis=1, inplace=True, level='type')
    child = cross(parents.iloc[0], parents.iloc[1])
    child = mutate(child)
    return child


def roulette_selection(parents_population):
    inverse_ranking = 1 / parents_population['ratings']
    selection_probs = inverse_ranking / inverse_ranking.sum()
    selected_parents = [parents_population.sample(n=2, weights=selection_probs, replace=False) for _ in
                        range(POPULATION - OLD_POPULATION)]
    return selected_parents


def generate_new_population(parents_population, pool):
    parents_pairs = roulette_selection(parents_population)
    children_population = pool.map(cross_and_mutate, parents_pairs)
    parents_population.drop('ratings', axis=1, inplace=True, level='type')
    if OLD_POPULATION == 0:
        new = pd.DataFrame(children_population, columns=parents_population.columns, index=range(POPULATION))
    else:
        new = pd.concat([pd.DataFrame(children_population), parents_population.iloc[:OLD_POPULATION]],
                        ignore_index=True)
    return new


def save_data(chromosome: pd.Series):
    directory = f"{FUZZY_TYPE}-{MEMBERSHIP_COUNT}-{RANKING_CAPACITY}-{OLD_POPULATION}-{SEED_VALUE}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    chromosome.to_pickle(os.path.join(directory, "chromosome.pkl"))
    with open(os.path.join(directory, f"data.txt"), "w") as file:
        file.write(f"Chromosome:\n{chromosome}\n\n")
        file.write(f"POPULATION: {POPULATION}\n")
        file.write(f"FUZZY_TYPE: {FUZZY_TYPE}\n")
        file.write(f"MEMBERSHIP_COUNT: {MEMBERSHIP_COUNT}\n")
        file.write(f"RANKING_CAPACITY: {RANKING_CAPACITY}\n")
        file.write(f"OLD_POPULATION: {OLD_POPULATION}\n")
        file.write(f"Random Seed: {SEED_VALUE}\n")
        file.write(f"Deffuzification Method: {output.defuzzify_method}\n")


def optimize(population: pd.DataFrame, count, limit):
    best = population.iloc[0]
    with Pool() as pool:
        while count < limit:
            best = population.iloc[0] if best['ratings'].values[0] > population.iloc[0]['ratings'].values[0] else best
            print(best)
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
    return best


if __name__ == '__main__':
    input_data = input_csv('output.csv')
    first_values = [float(row.split(',')[0]) for row in input_data]
    second_values = [float(row.split(',')[1]) for row in input_data]
    output_values = [float(row.split(',')[2].strip()) for row in input_data]

    population = build_starting_population(input_values=(first_values, second_values), outputs=output_values)

    count = 0
    limit = 0
    #best_chromosome = optimize(population, count, limit)
    correct_chromosome(population.iloc[0])
    # draw_best(best_chromosome)
    # save_data(best_chromosome)
