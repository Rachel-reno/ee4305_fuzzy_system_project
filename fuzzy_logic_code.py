import matplotlib.pyplot as plt
import numpy as np
import argparse

# Function to visualize the fuzzy sets in graphical manner
def plot_fuzzy_sets(dict_used, title, xlabel, ylabel):
    age_dict = {'very young':[[0,1],[0,1],[10,1],[15,0]],'young':[[10,0],[15,1],[25,1],[30,0]],'middle aged':[[25,0],[30,1],[60,1],[70,0]],'old':[[60,0],[70,1],[80,1],[85,0]],'very old':[[80,0],[85,1],[120,1],[120,1]]}
    hr_dict = {'low':[[0,1],[0,1],[40,1],[60,0]], 'normal':[[40,0],[60,1],[100,1],[140,0]], 'high':[[100,0],[140,1],[200,1],[200,1]]}
    health_risk_dict = {'low':[[0,1],[0,1],[0.3,1],[0.4,0]], 'medium':[[0.3,0],[0.4,1],[0.6,1],[0.7,0]],'high':[[0.6,0],[0.7,1],[1,1],[1,1]]}
    plt.figure()
    dict_used = dict_used
    for keys in dict_used:
        if keys != 'low':
            continue
        params = dict_used[keys]
        x,y = zip(*params)
        plt.plot(x,y,label=keys)
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Uncomment to use accordingly
    # plt.xticks([0.3,0.4,0.6,0.7],[0.3,0.4,0.6,0.7]) # Used for health risk
    # plt.xticks([10,15,25,30,60,70,80,85,120],[10,15,25,30,60,70,80,85,120]) # Used for age
    # plt.xticks([40,60,100,140,200],[40,60,100,140,200]) # Used for heart rate
    plt.yticks([0.0,1.0],[0.0,1.0])
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(fontsize = 'small', loc='center right')
    plt.show()
# plot_fuzzy_sets(age_dict, 'Plot of fuzzy sets of age','Age','Degree of Membership') # Change the parameters accordingly

# Calculate the membership of an input
def calc_degree(x,a,b,c,d):
    if x < a or x > d:
        return 0
    elif a <= x <= b:
        return ((x-a)/(b-a))
    elif b <= x <= c:
        return 1
    elif c <= x <= d:
        return ((d-x)/(d-c))

# Mapping fuzzy output based on particular age and heart rate input
def fuzzy_inference(age_input, hr_input):
    age_dict = {'very young':[0,0,10,15],'young':[10,15,25,30],'middle aged':[25,30,60,70],'old':[60,70,80,85],'very old':[80,85,120,120]}
    age_membership = {}
    hr_dict = {'low':[0,0,40,60], 'normal':[40,60,100,140], 'high':[100,140,200,200]}
    hr_membership = {}
    rules = {'very young low':'high', 'very young normal':'low','very young high':'low',\
        'young low':'high','young normal':'low','young high':'medium',\
        'middle aged low':'high','middle aged normal':'low','middle aged high':'high',\
        'old low':'medium','old normal':'low','old high':'high',\
        'very old low':'medium','very old normal':'low','very old high':'high'}
    health_risk_dict = {'low':[0,0,0.3,0.4], 'medium':[0.3,0.4,0.6,0.7],'high':[0.6,0.7,1,1]}
    for keys in age_dict:
        params = age_dict[keys]
        a,b,c,d = params[0],params[1],params[2],params[3]
        membership_degree = calc_degree(age_input,a,b,c,d)
        if membership_degree != 0:
            age_membership[keys] = membership_degree
    for keys in hr_dict:
        params = hr_dict[keys]
        a,b,c,d = params[0],params[1],params[2],params[3]
        membership_degree = calc_degree(hr_input,a,b,c,d)
        if membership_degree != 0:
            hr_membership[keys] = membership_degree
    low_eval, med_eval, high_eval = [],[],[]
    for age_keys in age_membership:
        for hr_keys in hr_membership:
            rule_key = age_keys + ' ' + hr_keys
            eval_key = rules[rule_key]
            min_degree = min(age_membership[age_keys],hr_membership[hr_keys])
            if eval_key == 'low':
                low_eval.append(min_degree)
            elif eval_key == 'medium':
                med_eval.append(min_degree)
            elif eval_key == 'high':
                high_eval.append(min_degree)
    if len(low_eval) == 0:
        low_output = 0
    else:
        low_output = max(low_eval)
    if len(med_eval) == 0:
        med_output = 0
    else:
        med_output = max(med_eval)
    if len(high_eval) == 0:
        high_output = 0
    else:
        high_output = max(high_eval)
    return low_output,med_output,high_output

# Calculate the output and visualize the resulting output plot
def defuzzify(low_output,med_output,high_output):
    # Defuzzification is done using discrete optimization with 5 sample points (0.2,0.4,0.6,0.8,1.0)
    crisp_output = (0.2 * low_output + 0.4 * med_output + 0.6 * med_output + 0.8 * high_output + 1.0 * high_output)/(low_output+2*med_output+2*high_output)
    health_risk_dict = {'low':[0,0,0.3,0.4], 'medium':[0.3,0.4,0.6,0.7],'high':[0.6,0.7,1,1]}
    low_coord, med_coord, high_coord = [],[],[]
    plt.figure()
    for keys in health_risk_dict:
        params = health_risk_dict[keys]
        a,b,c,d = params[0],params[1],params[2],params[3]
        if keys == 'low':
            low_coord.append([a,0])
            low_coord.append([b,low_output])
            c = d- low_output*(d-c)
            low_coord.append([c,low_output])
            low_coord.append([d,0])
            x,y = zip(*low_coord)
            plt.plot(x,y,label=keys)
        elif keys == 'medium':
            med_coord.append([a,0])
            b = a + med_output*(b-a)
            med_coord.append([b,med_output])
            c = d - med_output*(d-c)
            med_coord.append([c,med_output])
            med_coord.append([d,0])
            x,y = zip(*med_coord)
            plt.plot(x,y,label=keys)
        elif keys == 'high':
            high_coord.append([a,0])
            b = a + high_output*(b-a)
            high_coord.append([b,high_output])
            high_coord.append([c,high_output])
            high_coord.append([d,0])
            x,y = zip(*high_coord)
            plt.plot(x,y,label=keys)
    max_ylim = max(low_output,med_output,high_output)
    plt.xlim(0,1.1)
    plt.ylim(0,max_ylim+0.1)
    plt.title('Plot of fuzzy sets of health risk (output)')
    plt.xlabel('Health Risk')
    plt.ylabel('Degree of membership')
    plt.xticks([0.3,0.4,0.6,0.7],[0.3,0.4,0.6,0.7])
    # plt.xticks([10,15,25,30,60,70,80,85,120],[10,15,25,30,60,70,80,85,120])
    # plt.xticks([40,60,100,140,200],[40,60,100,140,200])
    plt.yticks([0.0,low_output,med_output,high_output],[0.0,low_output,med_output,high_output])
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(fontsize = 'small', loc='upper right')
    plt.show()
    return crisp_output

def run_fuzzy_logic(age_input, hr_input):
    age_input, hr_input = float(age_input), float(hr_input)
    low_output,med_output,high_output = fuzzy_inference(age_input,hr_input)
    print(defuzzify(low_output,med_output,high_output))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fuzzy system logic")
    parser.add_argument('--age', required=True, help='age input from werable')
    parser.add_argument('--hr',required=True, help='heart rate input from wearable')
    args = parser.parse_args()
    run_fuzzy_logic(args.age,args.hr)