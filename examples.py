import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Definiowanie zmiennych wejściowych i wyjściowych
temperatura = ctrl.Antecedent(np.arange(0, 41, 1), 'temperatura')  # Zakres od 0°C do 40°C
wilgotnosc = ctrl.Antecedent(np.arange(0, 101, 1), 'wilgotnosc')  # Zakres od 0% do 100%
predkosc_wentylatora = ctrl.Consequent(np.arange(0, 101, 1), 'predkosc_wentylatora')  # Zakres od 0% do 100%

# Definiowanie zbiorów rozmytych dla temperatury
temperatura['niska'] = fuzz.trimf(temperatura.universe, [0, 0, 20])
temperatura['srednia'] = fuzz.trimf(temperatura.universe, [10, 20, 30])
temperatura['wysoka'] = fuzz.trimf(temperatura.universe, [20, 40, 40])

# Definiowanie zbiorów rozmytych dla wilgotności
wilgotnosc['niska'] = fuzz.trimf(wilgotnosc.universe, [0, 0, 40])
wilgotnosc['srednia'] = fuzz.trimf(wilgotnosc.universe, [20, 50, 80])
wilgotnosc['wysoka'] = fuzz.trimf(wilgotnosc.universe, [60, 100, 100])

# Definiowanie zbiorów rozmytych dla prędkości wentylatora
predkosc_wentylatora['niska'] = fuzz.trimf(predkosc_wentylatora.universe, [0, 0, 40])
predkosc_wentylatora['srednia'] = fuzz.trimf(predkosc_wentylatora.universe, [20, 50, 80])
predkosc_wentylatora['wysoka'] = fuzz.trimf(predkosc_wentylatora.universe, [60, 100, 100])

# Definiowanie reguł rozmytych
rule1 = ctrl.Rule(temperatura['niska'] & wilgotnosc['niska'], predkosc_wentylatora['niska'])
rule2 = ctrl.Rule(temperatura['niska'] & wilgotnosc['srednia'], predkosc_wentylatora['niska'])
rule3 = ctrl.Rule(temperatura['niska'] & wilgotnosc['wysoka'], predkosc_wentylatora['srednia'])

rule4 = ctrl.Rule(temperatura['srednia'] & wilgotnosc['niska'], predkosc_wentylatora['niska'])
rule5 = ctrl.Rule(temperatura['srednia'] & wilgotnosc['srednia'], predkosc_wentylatora['srednia'])
rule6 = ctrl.Rule(temperatura['srednia'] & wilgotnosc['wysoka'], predkosc_wentylatora['wysoka'])

rule7 = ctrl.Rule(temperatura['wysoka'] & wilgotnosc['niska'], predkosc_wentylatora['srednia'])
rule8 = ctrl.Rule(temperatura['wysoka'] & wilgotnosc['srednia'], predkosc_wentylatora['wysoka'])
rule9 = ctrl.Rule(temperatura['wysoka'] & wilgotnosc['wysoka'], predkosc_wentylatora['wysoka'])

# Tworzenie systemu kontrolnego
predkosc_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
predkosc = ctrl.ControlSystemSimulation(predkosc_ctrl)

# Ustawianie wartości wejściowych
predkosc.input['temperatura'] = 25  # Temperatura 25°C
predkosc.input['wilgotnosc'] = 100  # Wilgotność 100%

# Obliczanie wartości wyjściowej
predkosc.compute()

# Wyświetlenie wyniku
print(f"Prędkość wentylatora: {predkosc.output['predkosc_wentylatora']:.2f}%")

# Wyświetlenie funkcji przynależności
temperatura.view(sim=predkosc)
wilgotnosc.view(sim=predkosc)
predkosc_wentylatora.view(sim=predkosc)

temperatura_value = 25
niska_value = fuzz.interp_membership(temperatura.universe, temperatura['niska'].mf, temperatura_value)
srednia_value = fuzz.interp_membership(temperatura.universe, temperatura['srednia'].mf, temperatura_value)
wysoka_value = fuzz.interp_membership(temperatura.universe, temperatura['wysoka'].mf, temperatura_value)

# Wypisanie wyników
print(f"Funkcja przynależności dla temperatury {temperatura_value}°C:")
print(f"niska: {niska_value:.2f}")
print(f"srednia: {srednia_value:.2f}")
print(f"wysoka: {wysoka_value:.2f}")