import pickle
from utils import *
directory = f"{FUZZY_TYPE}-{MEMBERSHIP_COUNT}"


with open(f'{directory}/chromosome.pkl', 'rb') as file:
    data = pickle.load(file)

print(data)
draw_best(data, fuzzy_input_1, fuzzy_input_2, fuzzy_output)
