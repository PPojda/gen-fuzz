import random
from utils import *
random.seed = 43


def create_membership_functions(fuzzy_input_1, fuzzy_input_2, fuzzy_output):
    if MEMBERSHIP_COUNT == 3:
        if FUZZY_TYPE == 3:
            fuzzy_input_1['low'] = fuzz.trimf(fuzzy_input_1.universe, [0, 30, 50])
            fuzzy_input_1['medium'] = fuzz.trimf(fuzzy_input_1.universe, [20, 50, 80])
            fuzzy_input_1['high'] = fuzz.trimf(fuzzy_input_1.universe, [50, 70, 100])

            fuzzy_input_2['low'] = fuzz.trimf(fuzzy_input_2.universe, [0, 30, 50])
            fuzzy_input_2['medium'] = fuzz.trimf(fuzzy_input_2.universe, [20, 50, 80])
            fuzzy_input_2['high'] = fuzz.trimf(fuzzy_input_2.universe, [50, 70, 100])

            fuzzy_output['low'] = fuzz.trimf(fuzzy_output.universe, [0, 30, 50])
            fuzzy_output['medium'] = fuzz.trimf(fuzzy_output.universe, [20, 50, 80])
            fuzzy_output['high'] = fuzz.trimf(fuzzy_output.universe, [50, 70, 100])
        elif FUZZY_TYPE == 4:
            fuzzy_input_1['low'] = fuzz.trapmf(fuzzy_input_1.universe, [0, 0, 20, 40])
            fuzzy_input_1['medium'] = fuzz.trapmf(fuzzy_input_1.universe, [30, 50, 50, 70])
            fuzzy_input_1['high'] = fuzz.trapmf(fuzzy_input_1.universe, [60, 80, 100, 100])

            fuzzy_input_2['low'] = fuzz.trapmf(fuzzy_input_2.universe, [0, 0, 20, 40])
            fuzzy_input_2['medium'] = fuzz.trapmf(fuzzy_input_2.universe, [30, 50, 50, 70])
            fuzzy_input_2['high'] = fuzz.trapmf(fuzzy_input_2.universe, [60, 80, 100, 100])

            fuzzy_output['low'] = fuzz.trapmf(fuzzy_output.universe, [0, 0, 20, 40])
            fuzzy_output['medium'] = fuzz.trapmf(fuzzy_output.universe, [30, 50, 50, 70])
            fuzzy_output['high'] = fuzz.trapmf(fuzzy_output.universe, [60, 80, 100, 100])
        else:
            raise ValueError("FUZZY_TYPE must be 3 or 4")
    elif MEMBERSHIP_COUNT == 4:
        if FUZZY_TYPE == 3:
            fuzzy_input_1['low'] = fuzz.trimf(fuzzy_input_1.universe, [0, 20, 40])
            fuzzy_input_1['medium_low'] = fuzz.trimf(fuzzy_input_1.universe, [20, 40, 60])
            fuzzy_input_1['medium_high'] = fuzz.trimf(fuzzy_input_1.universe, [40, 60, 80])
            fuzzy_input_1['high'] = fuzz.trimf(fuzzy_input_1.universe, [60, 80, 100])

            fuzzy_input_2['low'] = fuzz.trimf(fuzzy_input_2.universe, [0, 20, 40])
            fuzzy_input_2['medium_low'] = fuzz.trimf(fuzzy_input_2.universe, [20, 40, 60])
            fuzzy_input_2['medium_high'] = fuzz.trimf(fuzzy_input_2.universe, [40, 60, 80])
            fuzzy_input_2['high'] = fuzz.trimf(fuzzy_input_2.universe, [60, 80, 100])

            fuzzy_output['low'] = fuzz.trimf(fuzzy_output.universe, [0, 20, 40])
            fuzzy_output['medium_low'] = fuzz.trimf(fuzzy_output.universe, [20, 40, 60])
            fuzzy_output['medium_high'] = fuzz.trimf(fuzzy_output.universe, [40, 60, 80])
            fuzzy_output['high'] = fuzz.trimf(fuzzy_output.universe, [60, 80, 100])
        elif FUZZY_TYPE == 4:
            fuzzy_input_1['low'] = fuzz.trapmf(fuzzy_input_1.universe, [0, 0, 20, 40])
            fuzzy_input_1['medium_low'] = fuzz.trapmf(fuzzy_input_1.universe, [20, 30, 40, 60])
            fuzzy_input_1['medium_high'] = fuzz.trapmf(fuzzy_input_1.universe, [40, 60, 70, 80])
            fuzzy_input_1['high'] = fuzz.trapmf(fuzzy_input_1.universe, [60, 80, 100, 100])

            fuzzy_input_2['low'] = fuzz.trapmf(fuzzy_input_2.universe, [0, 0, 20, 40])
            fuzzy_input_2['medium_low'] = fuzz.trapmf(fuzzy_input_2.universe, [20, 30, 40, 60])
            fuzzy_input_2['medium_high'] = fuzz.trapmf(fuzzy_input_2.universe, [40, 60, 70, 80])
            fuzzy_input_2['high'] = fuzz.trapmf(fuzzy_input_2.universe, [60, 80, 100, 100])

            fuzzy_output['low'] = fuzz.trapmf(fuzzy_output.universe, [0, 0, 20, 40])
            fuzzy_output['medium_low'] = fuzz.trapmf(fuzzy_output.universe, [20, 30, 40, 60])
            fuzzy_output['medium_high'] = fuzz.trapmf(fuzzy_output.universe, [40, 60, 70, 80])
            fuzzy_output['high'] = fuzz.trapmf(fuzzy_output.universe, [60, 80, 100, 100])
        else:
            raise ValueError("FUZZY_TYPE must be 3 or 4")
    else:
        raise ValueError("MEMBERSHIP_COUNT must be 3 or 4")


def generate_input():
    first = np.random.random() * 100
    second = np.random.random() * 100
    return first, second


def evaluate_output(input_1, input_2, sim):
    sim.input['first_input'] = input_1
    sim.input['second_input'] = input_2
    sim.compute()
    return sim.output['output']


def view(output_sim, fuzzy_input_1, fuzzy_input_2, fuzzy_output):
    output_sim.input['first_input'] = 50
    output_sim.input['second_input'] = 50
    output_sim.compute()

    fuzzy_input_1.view()
    fuzzy_input_2.view()
    fuzzy_output.view(sim=output_sim)


if __name__ == '__main__':
    create_membership_functions(fuzzy_input_1, fuzzy_input_2, fuzzy_output)
    output_sim = simulate(fuzzy_input_1, fuzzy_input_2, fuzzy_output)
    view(output_sim, fuzzy_input_1, fuzzy_input_2, fuzzy_output)

    outputs = set()
    for i in range(2000):
        fuzzy_input_1, fuzzy_input_2 = generate_input()
        output_value = evaluate_output(fuzzy_input_1, fuzzy_input_2, output_sim)
        outputs.add(output_value)
        with open('output.csv', 'a') as f:
            f.write("{},{},{}\n".format(fuzzy_input_1, fuzzy_input_2, output_value))
    print(len(outputs))
