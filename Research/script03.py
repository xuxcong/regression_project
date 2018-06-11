from model_test import ModelBase,summary
import numpy as np
import pandas as pd
from pdb import set_trace
import matplotlib.pyplot as plt

f = open("../Data/train.csv")
df = pd.read_csv(f)
f.close()

sexes = df["Sex"]
ages = df["Age"]
survivals = df["Survived"]
pclasses = df["Pclass"]
cabins = df["Cabin"]
had_cabin_records = pd.Series([isinstance(x,str) for x in cabins])

cabins = cabins[had_cabin_records]
survivals = survivals[had_cabin_records]

def get_deck(cabin):
    tokens = cabin.split()
    deck = tokens[0][0]
    return deck

def get_number(cabin):
    tokens = cabin.split()
    for i in range(len(tokens)):
        if len(tokens[i])<=1:
            continue
        else:
            number = tokens[i][1:]
            return int(number)
    return np.nan


decks = pd.Series(list(map(lambda x:get_deck(x),cabins)),index=survivals.index)
deck_codes = list(set(decks))

numbers = pd.Series(list(map(lambda x:get_number(x),cabins)),index=survivals.index)
# print(numbers)
# set_trace()
# print(list(np.isnan(numbers)).index(True))
# print(np.any(np.isnan(numbers)))
max_number = max(numbers)
min_number = min(numbers)
n=5
cuts = np.linspace(min_number,max_number,n+1)
# set_trace()
survive_ratios_deck = {}
deck_denominators = {}

survive_ratios_number = []
number_denominators = []


for deck in deck_codes:
    idx = decks==deck
    # set_trace()
    survive_ratios_deck[deck]=np.nanmean(survivals[idx])
    deck_denominators[deck] = sum(idx)

for i in range(n):
    min_num = cuts[i]
    max_num = cuts[i+1]
    if i<n-1:
        idx = (numbers>=min_num)&(numbers<max_num)
    else:
        idx = (numbers>=min_num)&(numbers<=max_num)
    survive_ratios_number.append(np.nanmean(survivals[idx]))
    number_denominators.append(len(survivals[idx]))

print(deck_denominators)
print(survive_ratios_deck)

# plt.bar(cuts[0:n],survive_ratios_number,align="edge",width=(max_number-min_number)/(n+1))
# plt.show()
# print(number_denominators)

fig = plt.figure()
ax = fig.add_plot(projection="3d")

deck_codes_list = list(deck_codes)
top = np.zeros((len(deck_codes),n))
for i in range(len(deck_codes)):
    for j in range(n):
        deck = deck_codes_list[i]
        min_num = cuts[j]
        max_num = cuts[j+1]
        if i<n-1:
            idx = (numbers>=min_num)&(numbers<max_num)
        else:
            idx = (numbers>=min_num)&(numbers<=max_num)
        idx = idx & (decks==deck)
        top[i,j]=np.nanmean(survivals[idx])



x = np.linspace(1,len(deck_codes),len(deck_codes))
y = cuts[0:n]

# ax.bar3d(x,y,bottom,width,depth,top,shade=True)
ax.set_title("")
plt.show()