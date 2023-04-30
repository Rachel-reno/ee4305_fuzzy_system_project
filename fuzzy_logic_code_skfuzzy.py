import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
import argparse

def health_prediction(age_input, heart_rate_input, show_fuzzy_sets=False, decision_making = False):
    # Defining fuzzy sets 
    age_input, heart_rate_input = float(age_input), float(heart_rate_input)
    age = ctrl.Antecedent(np.arange(0,121,1), 'age')
    heart_rate = ctrl.Antecedent(np.arange(0,201,1), 'heart rate')
    health_risk = ctrl.Consequent(np.arange(0,1.1,0.1),'health risk')
    age['very young'] = fuzz.trapmf(age.universe,[0,0,10,15])
    age['young'] = fuzz.trapmf(age.universe,[10,15,30,35])
    age['middle aged'] = fuzz.trapmf(age.universe,[30,35,60,70])
    age['old'] = fuzz.trapmf(age.universe,[60,70,80,85])
    age['very old'] = fuzz.trapmf(age.universe,[80,85,120,120])
    heart_rate['low'] = fuzz.trapmf(heart_rate.universe,[0,0,40,60])
    heart_rate['normal'] = fuzz.trapmf(heart_rate.universe,[40,60,100,140])
    heart_rate['high'] = fuzz.trapmf(heart_rate.universe,[100,140,200,200])
    health_risk['low'] = fuzz.trapmf(health_risk.universe,[0,0,0.3,0.4])
    health_risk['medium'] = fuzz.trapmf(health_risk.universe,[0.3,0.4,0.6,0.7])
    health_risk['high'] = fuzz.trapmf(health_risk.universe,[0.6,0.7,1,1])
    # Show the fuzzy sets
    if show_fuzzy_sets:
        age.view()
        plt.show()
        heart_rate.view()
        plt.show()
        health_risk.view()
        plt.show()

    # Provide rules
    rule1 = ctrl.Rule(age['very young'] & heart_rate['low'], health_risk['high'])
    rule2 = ctrl.Rule(age['very young'] & heart_rate['normal'], health_risk['low'])
    rule3 = ctrl.Rule(age['very young'] & heart_rate['high'], health_risk['low'])
    rule4 = ctrl.Rule(age['young'] & heart_rate['low'], health_risk['high'])
    rule5 = ctrl.Rule(age['young'] & heart_rate['normal'], health_risk['low'])
    rule6 = ctrl.Rule(age['young'] & heart_rate['high'], health_risk['medium'])
    rule7 = ctrl.Rule(age['middle aged'] & heart_rate['low'], health_risk['high'])
    rule8 = ctrl.Rule(age['middle aged'] & heart_rate['normal'], health_risk['low'])
    rule9 = ctrl.Rule(age['middle aged'] & heart_rate['high'], health_risk['high'])
    rule10 = ctrl.Rule(age['old'] & heart_rate['low'], health_risk['medium'])
    rule11 = ctrl.Rule(age['old'] & heart_rate['normal'], health_risk['low'])
    rule12 = ctrl.Rule(age['old'] & heart_rate['high'], health_risk['high'])
    rule13 = ctrl.Rule(age['very old'] & heart_rate['low'], health_risk['medium'])
    rule14 = ctrl.Rule(age['very old'] & heart_rate['normal'], health_risk['low'])
    rule15 = ctrl.Rule(age['very old'] & heart_rate['high'], health_risk['high'])

    # Conduct the fuzzy logic system
    health_pred_ctrl = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,rule10,rule11,rule12,rule13,rule14,rule15])
    health_pred = ctrl.ControlSystemSimulation(health_pred_ctrl)
    health_pred.input['age'] = age_input
    health_pred.input['heart rate'] = heart_rate_input
    health_pred.compute()
    output_results = health_pred.output['health risk']

    # Plot the decision making
    if decision_making:
        health_risk.view(sim=health_pred)
        plt.show()
    print(output_results)
    return output_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fuzzy system logic")
    parser.add_argument('--age', required=True, help='age input from werable')
    parser.add_argument('--hr',required=True, help='heart rate input from wearable')
    parser.add_argument('--show_mf', help='input True if fuzzy set plots will be plotted')
    parser.add_argument('--show_decision', help='show results in graph')
    args = parser.parse_args()
    health_prediction(args.age,args.hr,args.show_mf,args.show_decision)