# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB

""" it will import the math library for mathematical operation """

import math


N = 10000   # number of iteration
d = 10      # number of arms (distribution)
ads_selected = [] # it is infinite array that possess a ad selected (value) in a iteration (index)
numbers_of_selections = [0] * d  # it is array of d index possess number of time i(th) arm selected 
sums_of_rewards = [0] * d # it count number of 1 (reward) if arm selected correct
total_reward = 0 # total reward count the total number of reward af all arms

# it will iterate the loop 10,000 ( exploration )

for n in range(0, N): 
    ad = 0
    max_upper_bound = 0
    
    # it will iterate the loop for selection of arm in d arms
    
    for i in range(0, d):
        
        # apply the algorithm if all arms expoited once
        
        if (numbers_of_selections[i] > 0):
            
        # it will calculate the average reward and delta i to calculate upper bound
            
            
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        
        # it is applied to expolit once a arm
        
        else:
            upper_bound = 1e400
            
        # if upper bound > max upper bound 
        '''
            -> then variate the value of max upper bound 
            -> get the i(th) arm to exploit the arm
        
        '''
        
        
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    
    '''
        it will append the arm selected in infinite array
        it will increase the number of time the arm selected by one
        it will reward if the seleted arm correct or punish if not
        it will calculate the sum of reward for each arm
        it will calculate total reward
    '''
    
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Visualising the results
    
# it will make a histogram (bar graph) of the number of selection vs i(th) ad
    
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()