from plackettluce import *
import numpy as np
from matplotlib import pyplot as plt

# Fix N (#voters) and m (#alternatives)
N = 100
m = 6

# sample gamma for all voters
# gamma_all[i] will be the PL parameters for voter i
gamma_all = np.zeros([N,m])

for i in range(N):
    gamma = np.random.rand(m)
    gamma /= np.sum(gamma) # normalize sum to 1.0 (not needed for Dirichlet)
    gamma_all[i] = gamma

# get average PL parameter
gamma_mean = np.mean(gamma_all, axis = 0)

#%% voting rules function

def pluraliy(ballots):
    N = len(ballots)
    m = len(ballots[0])
    
    #calculate plurality score for each alternative
    score = np.zeros(m)
    for i in range(N):
        score[int(ballots[i][0])] += 1
    #return plurality winner, the one with highest plurality score
    return np.argmax(score)

def Borda(ballots):
    N = len(ballots)
    m = len(ballots[0])
    
    #calculate Borda score for each alternative
    score = np.zeros(m)
    for i in range(N):
        for j in range(m):
            score[int(ballots[i][j])] += (m-j-1)
    #return Borda winner, the one with highest Borda score
    return np.argmax(score)

#%% calculate winner distribution for all profiles
T = 10000 #number of samples
winner_1 = np.zeros(m)
for t in range(T):
    ballots = np.zeros([N,m])
    for i in range(N):
        #draw a vote from each voter's PL model
        ballots[i] = draw_pl_vote(m,gamma_all[i])
    #calculate winner based on the votes
    winner_1[Borda(ballots)] += 1

#%% calculate winner distribution for mean_profile
    
T = 10000 #number of samples
winner_2 = np.zeros(m)
for t in range(T):
    # number of samples we take for each vote
    # we just equal it to the number of voters for now
    # both T, T_2 might need further tweaking
    T_2 = N 
    ballots = np.zeros([T_2,m])
    #draw  T_2 vote from the mean PL model
    for i in range(T_2):
        ballots[i] = draw_pl_vote(m,gamma_mean)
    #calculate winner based on the votes
    winner_2[Borda(ballots)] += 1
    
#%% calculate randomized voting rule result
# for now, only Borda

def randomized_borda_score(gamma_all):
    '''
        Randomized borda result would be a function of the parameters only
            no need to take sample votes
    '''
    m = len(gamma_all[0])
    n = len(gamma_all)    
    Borda_score = np.zeros(m)
    # randomized borda_score for any candidate a_j
    #   Pr(a_j) \isprop \sum_{voter=0}^{N-1} \sum_{j'!=j} (gamma(a_j)/(gamma(a_j)+gamma(a_j'))) 
    for i in range(m):
        for j in range(m):
            if(i==j):
                continue
            for user in range(n):
                Borda_score[i] += gamma_all[user][i] / (gamma_all[user][i] + gamma_all[user][j])
    return Borda_score

winner_3 = randomized_borda_score(gamma_all)

#%%
def print_pretty(winner):
    np.set_printoptions(precision=3)
    winner = np.array(winner)
    print(winner / np.sum(winner))
    print(np.argsort(winner))
    
print_pretty(winner_1)
print_pretty(winner_2)
print_pretty(winner_3)