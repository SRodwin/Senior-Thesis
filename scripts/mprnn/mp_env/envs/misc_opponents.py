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
        self.beta = 2
        self.beta2 = 10
        self.omega = 0.5
        self.lr_1 = 0.2 # 0.2, 0.5, 0.8
        self.lr_2 = 0.2 # 0.2, 0.5, 0.8
        self.p_star = 0.5
        self.q_ss = 0.5 # This is the iniial q** as we are assuming that initial choice is random 
        
        
        
    def __str__(self):
        return 'InfluenceOpponent(lr_1={}, lr_2={})'.format(self.lr_1, self.lr_2)
    
    
    def step(self, choice, rew):
        opp_choice = 0
        #generate opponent's choice at t based on p_star and payoff matrix
        v_opp_right = 1 - self.p_star
        v_opp_left = self.p_star

        # choice output of opponent at time step t
        #  -1 <= v_opp_right-v_opp_left <= 1
        # with beta==1, 0.269<=opp_PR<=0.7311
        opp_PR = 1 / (1 + np.exp(-self.beta * (v_opp_right - v_opp_left)))

        if np.random.rand() < opp_PR:
            opp_choice = 1

        #updating p_star based on agent's probability of right choice
        update_p_star_via_choice = choice[-1] - self.p_star  #[-1 1]

        
        #update q_ss (agent's estimate of opponent's probability of right)
        update_q_ss = self.q_ss + self.lr_2 * (opp_choice - self.q_ss)  #[-1 1]
        
        # use payoff matrix to calcuate agent's values of left and right choice
        # based on the estimated probability of opponent's right choice
        v_ag_right_via_qss1 = self.q_ss
        v_ag_left_via_qss1 = 1 - self.q_ss
        #agent's p_star based on q_ss at the previous step
        p_star_via_qss1 = 1 / (1 + np.exp(-self.beta2 * (v_ag_right_via_qss1 - v_ag_left_via_qss1)))
                    
        v_ag_right_via_qss2 = update_q_ss
        v_ag_left_via_qss2 = 1 - update_q_ss
        #agent's p_star based on q_ss at the current step

        # agent's p_star based on q_ss at the current step
        p_star_via_qss2 = 1 / (1 + np.exp(-self.beta2 * (v_ag_right_via_qss2 - v_ag_left_via_qss2)))

        # estimated change in p_star due to the change in q_ss
        # from the previous time step
        # second-order belief-based update of P_star
        delta_p_star_via_qss = p_star_via_qss2 - p_star_via_qss1 #[-0.9 0.9]
        wavg = self.omega*update_p_star_via_choice+(1-self.omega)*delta_p_star_via_qss
        p_star_old = self.p_star


        self.p_star = self.p_star+self.lr_1*wavg
        if self.p_star < 0:
            self.p_star = 0 
        if self.p_star > 1:
            self.p_star = 1
        self.q_ss = update_q_ss
        
        return opp_choice, p_star_old,None 
      
         

        
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

        

