import numpy as np
from random import random
from scipy.stats import binom_test

#this file contains classes of opponents to be played against in the matching pennies game
#they must implement the step function
rng = np.random.default_rng()
class Opponent():
    def __init__(self,**kwargs):
        self.opp_kwargs = kwargs
        self.opponent_action = rng.choice([0,1])
        self.pchooseright = 0.5
        self.biasinfo = None
        self.bias = kwargs.get('bias',0)
        self.act_hist = [self.opponent_action]
    def get_last_saved(self):
        return self.opponent_action, self.pchooseright,self.biasinfo
    def step(self,choice,rew):
        '''
        choice is the agent choice (the deep RL model) and its reward
        '''
        raise NotImplementedError


class MatchingPenniesOpponent(Opponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
        self.alpha = 0.1
        self.pComputerRight = 0.5
        self.bias = 0
        self.biasDepth = -1
        self.maxDev = 0
        self.maxdepth = 3
    
    def __str__(self):
        return 'MatchingPenniesOpponent(alpha={})'.format(self.alpha)

    def step(self, choice, rew):

        if choice is None:
            return (1 if random() < self.pComputerRight else 0), self.pComputerRight, [0,-1,0,0,const_bias]
        data = np.array(choice)
        choice, rew = np.array(choice) + 1, np.array(rew) + 1 #recode as 1/2
        choice = np.append(choice,None)
        rew = np.append(rew,None)

        for depth in range(self.maxdepth):
            if data.size < depth + 1:
                continue
            if depth == 0: #count all right choices to check overall side bias
                countN = len(choice)
                countRight = np.sum(data)
            else: #look for bias on trials where recent choice history was the same
                histseq = np.zeros(len(choice))
                for trial in range(depth+1,len(choice)):
                    seq = 0
                    for currdepth in range(1,depth):
                        seq += choice[trial-currdepth]* (10**(currdepth-1))
                    histseq[trial] = seq
                idx = np.where(histseq == histseq[-1])[:-1]
                if not idx:
                    continue
                countRight = np.sum((data[idx]))
                countN = len(idx)

            pRightBias = 1 - binom_test(countRight-1,countN,0.5) #p(X>=x)
            pLeftBias = binom_test(countRight,countN,0.5) #p(X<=x)
            pDeviation = countRight / countN - 0.5 #p(Right)

            if pRightBias < self.alpha or pLeftBias < self.alpha and \
            np.abs(pDeviation) > np.abs(self.maxDev):
                    self.maxDev = pDeviation
                    self.bias = 1 if self.maxDev < 0 else 2
                    self.biasDepth = depth
                    self.pComputerRight = 0.5 - self.maxDev


        for depth in range(self.maxdepth):
            if data.size < depth+1:
                continue
            chistseq = np.empty(len(choice))
            rhistseq = np.empty(len(rew))

            for trial in range(depth+1,len(choice)):
                cseq, rseq = 0,0
                for currdepth in range(1,depth):
                    cseq += choice[trial-currdepth] * 10 ** (currdepth-1)
                    rseq += rew[trial-currdepth] * 10 ** (currdepth-1)
                chistseq[trial] = cseq
                rhistseq[trial] = rseq
            idx = np.where(np.logical_and(chistseq == chistseq[-1], rhistseq == rhistseq[-1]))
            idx = idx[:-1]
            if not idx:
                continue
            countRight = np.sum((data[idx]))
            countN = len(idx)
            pRightBias = 1 - binom_test(countRight-1,countN,0.5) #p(X>=x)
            pLeftBias = binom_test(countRight,countN,0.5) #p(X<=x)
            pDeviation = countRight / countN - 0.5 #p(Right)

            if pRightBias < self.alpha or pLeftBias < self.alpha and \
                np.abs(pDeviation) > np.abs(self.maxDev):
                    self.maxDev = pDeviation
                    self.bias = 1 if self.maxDev < 0 else 2
                    self.biasDepth = depth
                    self.pComputerRight = 0.5 - self.maxDev

        biasInfo = [self.bias,self.biasDepth,self.maxDev]
        #might need to flip 0 and 1!
        computerChoice = 1 if random() < self.pComputerRight  else 0
        return computerChoice,self.pComputerRight,biasInfo

class InfluenceOpponent(Opponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
        self.lr_1 = 0.5 # 0.2, 0.5, 0.8
        self.lr_2 = 0.8 # 0.2, 0.5, 0.8
        self.q_ss = 0.5 # This is the iniial q** as we are assuming that initial choice is random 
        
        
    def __str__(self):
        return 'InfluenceOpponent(lr_1={}, lr_2={})'.format(self.lr_1, self.lr_2)
    
    
    def step(self, choice, rew):
        action = 0
        # Get Qt 
        Qt = self.act_hist[-1] if len(self.act_hist) > 0 else 0.5
        
        # Get q** 
        q_ss = self.q_ss
        
        # Calculate V(R) and V(L) for Qt and q** 
        V_R_Qt = (1 - Qt) * 1 + Qt * 0
        V_L_Qt = (1 - Qt) * 0 + Qt * 1
        V_R_q_ss = (1 - q_ss) * 0 + q_ss * 1
        V_L_q_ss = (1 - q_ss) * 1 + q_ss * 0
       
        # Calculate z 
        z1 = self.q_ss - (1 - self.q_ss)
        z2 = Qt - (1 - Qt)
        
        
        # Calculate soft max for Qt and q**
        soft_max_Qt = 1 / (1 + np.exp(-z2 * self.lr_2))
        soft_max_q_star = 1 / (1 + np.exp(-z1 * self.lr_1))
      
        # Calculate delta p* 
        delta_p_star =  soft_max_Qt - soft_max_q_star
        
        # Calculate T 
        T = self.lr_2 * delta_p_star
        
        # Calculate p_2*
        p_2_star = q_ss + self.lr_1 * (choice - q_ss) + T
        
        # Decide action 
        if np.random.rand() < p_2_star:
            action = 1
        
        # Update act_history and q** 
        self.act_hist.append(action)
        self.q_ss = q_ss + self.lr_1 * (Qt - q_ss)
        
        return action,self.q_ss,None 

        
class SoftmaxQlearn(Opponent):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.alpha = 0.8 # 0.2, 0.5, 0.8
        self.beta = 0.8 # 0.2, 0.5, 0.8
        self.Qs = np.zeros((2))
        self.max_choices = []

    def step(self,choice,rew):
        rew = 1 - rew #if opponent wins, agent loses--rew is agent reward, so flip bit
        self.RPEupdate(rew)
        opponent_action, pchooseright, biasinfo = self.action()
        self.opponent_action = opponent_action
        self.pchooseright = pchooseright
        self.biasinfo = biasinfo
        self.act_hist.append(opponent_action)
        return opponent_action,pchooseright,biasinfo

    def action(self):
        action_data = {'Qs':self.Qs,'bias':self.bias}
       
        self.ql, self.qr = self.Qs
        self.max_choices.append(np.argmax(self.Qs))
        
        #softmax to find probability
        self.pr = 1 / (1 + np.exp(- self.beta * (self.qr - self.ql)))
        
        if not self.pr:
            self.pr = 10 ** -10
        action = 1 if rng.random() < self.pr else 0 #random choice

        return action, self.pr, action_data

    def RPEupdate(self, reward):
        """Reward Prediction Error Calculation and Update of Q values"""
        Q_prev = self.Qs[self.act_hist[-1]] #q-value of the action just taken
        self.Qs[self.act_hist[-1]] = Q_prev + self.alpha * (reward - Q_prev)
    
    def __str__(self):
        return 'SoftmaxQlearn(alpha={}, beta={})'.format(self.alpha, self.beta)



class PatternBandit(Opponent):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self.pattern = [random.randint(0, 1) for i in range(8)] #fixed pattern to play
        self.plen = len(self.pattern)
        self.count = 0
        self.max_choices = []

    def __str__(self):
        return f'patternbandit with pattern {self.pattern}'

    def step(self,choice,rew):
        ind = self.count % self.plen #where in the pattern are we?
        action = int(self.pattern[ind])
        self.max_choices.append(action)
        self.count += 1 #step forward
        self.act_hist.append(action)
        self.opponent_action = action
        self.pchooseright = 1 if action else 0
        self.biasinfo = None
        return action, self.pchooseright,self.biasinfo

class MimicryPlayer(Opponent):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.n = 8
        self.acts_inframe = [] #we don't care about act_hist, but rather only the last n choices
    def __str__(self):
        return f"t-{self.n} mimicry"
    
    def step(self,choice,rew):
        self.acts_inframe.append(choice) 
        if len(self.acts_inframe) >= self.n: #if we have played more than n trials, play the opposite choice that the agent took
            return 1 - self.acts_inframe.pop(0),self.pchooseright,None
        else:
            return rng.choice([0,1]),self.pchooseright,None

        

