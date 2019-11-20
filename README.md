The plackettluce.py is from the prefpy library
The result_comparison currently compares the winner distributions from 3 approaches
1. sample complete preference profiles by sampling votes from each voter's distribution and calculate winner
2. create a summary model, sample all votes from that distribution and calculate winner
3. randomized voting rules

Remarks:
1. The top candidate with highest winning probability almost always agrees
2. Even the rank based on winning probability is mostly same between the three methods
3. The probabilities themselves, however vary greatly, with the greates "error" for randomized voting rules
