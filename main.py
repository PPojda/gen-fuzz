import random
from utils import *
random.seed = 43


def create_membership_functions():
    global input_1, input_2, output
    if MEMBERSHIP_COUNT == 3:
        if FUZZY_TYPE == 3:
            input_1['low'] = fuzz.trimf(input_1.universe, [0, 30, 50])
            input_1['medium'] = fuzz.trimf(input_1.universe, [20, 50, 80])
            input_1['high'] = fuzz.trimf(input_1.universe, [50, 70, 100])

            input_2['low'] = fuzz.trimf(input_2.universe, [0, 30, 50])
            input_2['medium'] = fuzz.trimf(input_2.universe, [20, 50, 80])
            input_2['high'] = fuzz.trimf(input_2.universe, [50, 70, 100])

            output['low'] = fuzz.trimf(output.universe, [0, 30, 50])
            output['medium'] = fuzz.trimf(output.universe, [20, 50, 80])
            output['high'] = fuzz.trimf(output.universe, [50, 70, 100])
        elif FUZZY_TYPE == 4:
            input_1['low'] = fuzz.trapmf(input_1.universe, [0, 0, 20, 40])
            input_1['medium'] = fuzz.trapmf(input_1.universe, [30, 50, 50, 70])
            input_1['high'] = fuzz.trapmf(input_1.universe, [60, 80, 100, 100])

            input_2['low'] = fuzz.trapmf(input_2.universe, [0, 0, 20, 40])
            input_2['medium'] = fuzz.trapmf(input_2.universe, [30, 50, 50, 70])
            input_2['high'] = fuzz.trapmf(input_2.universe, [60, 80, 100, 100])

            output['low'] = fuzz.trapmf(output.universe, [0, 0, 20, 40])
            output['medium'] = fuzz.trapmf(output.universe, [30, 50, 50, 70])
            output['high'] = fuzz.trapmf(output.universe, [60, 80, 100, 100])
        else:
            raise ValueError("FUZZY_TYPE must be 3 or 4")
    elif MEMBERSHIP_COUNT == 4:
        if FUZZY_TYPE == 3:
            input_1['low'] = fuzz.trimf(input_1.universe, [0, 0, 30])
            input_1['medium_low'] = fuzz.trimf(input_1.universe, [20, 40, 60])
            input_1['medium_high'] = fuzz.trimf(input_1.universe, [40, 60, 80])
            input_1['high'] = fuzz.trimf(input_1.universe, [70, 100, 100])

            input_2['low'] = fuzz.trimf(input_2.universe, [0, 0, 30])
            input_2['medium_low'] = fuzz.trimf(input_2.universe, [20, 40, 60])
            input_2['medium_high'] = fuzz.trimf(input_2.universe, [40, 60, 80])
            input_2['high'] = fuzz.trimf(input_2.universe, [70, 100, 100])

            output['low'] = fuzz.trimf(output.universe, [0, 0, 30])
            output['medium_low'] = fuzz.trimf(output.universe, [20, 40, 60])
            output['medium_high'] = fuzz.trimf(output.universe, [40, 60, 80])
            output['high'] = fuzz.trimf(output.universe, [70, 100, 100])
        elif FUZZY_TYPE == 4:
            input_1['low'] = fuzz.trapmf(input_1.universe, [0, 0, 20, 40])
            input_1['medium_low'] = fuzz.trapmf(input_1.universe, [20, 30, 40, 60])
            input_1['medium_high'] = fuzz.trapmf(input_1.universe, [40, 60, 70, 80])
            input_1['high'] = fuzz.trapmf(input_1.universe, [60, 80, 100, 100])

            input_2['low'] = fuzz.trapmf(input_2.universe, [0, 0, 20, 40])
            input_2['medium_low'] = fuzz.trapmf(input_2.universe, [20, 30, 40, 60])
            input_2['medium_high'] = fuzz.trapmf(input_2.universe, [40, 60, 70, 80])
            input_2['high'] = fuzz.trapmf(input_2.universe, [60, 80, 100, 100])

            output['low'] = fuzz.trapmf(output.universe, [0, 0, 20, 40])
            output['medium_low'] = fuzz.trapmf(output.universe, [20, 30, 40, 60])
            output['medium_high'] = fuzz.trapmf(output.universe, [40, 60, 70, 80])
            output['high'] = fuzz.trapmf(output.universe, [60, 80, 100, 100])
        else:
            raise ValueError("FUZZY_TYPE must be 3 or 4")
    else:
        raise ValueError("MEMBERSHIP_COUNT must be 3 or 4")


def generate_input():
    first = np.random.randint(0, 101)
    second = np.random.randint(0, 101)
    return first, second


def evaluate_output(input_1, input_2, sim):
    sim.input['first_input'] = input_1
    sim.input['second_input'] = input_2
    sim.compute()
    return sim.output['output']


def view():
    global output_sim, input_1, input_2, output
    output_sim.input['first_input'] = 50
    output_sim.input['second_input'] = 50
    output_sim.compute()

    input_1.view()
    input_2.view()
    output.view(sim=output_sim)


if __name__ == '__main__':
    create_membership_functions()
    output_sim = simulate()
    view()

    outputs = set()
    for i in range(1000):
        input_1, input_2 = generate_input()
        output_value = evaluate_output(input_1, input_2, output_sim)
        outputs.add(output_value)
        with open('output.csv', 'a') as f:
            f.write("{},{},{}\n".format(input_1, input_2, output_value))
    print(len(outputs))
