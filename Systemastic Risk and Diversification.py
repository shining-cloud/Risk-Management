# This project includes simulation of the effect of systemastic risk on the insurance policy (independent)  and diversification effect.

'''
Risk Exposure without Systemastic Risk

The insurance company should invest enough capital to cover the possible loss, so it's important for insurance companies to know how much capital it should prepare when the loss exceeds the expected loss to make the risk management. Meanwhile, we know that diversification will decrease the risk of portfolio intuitively. So in this part, we will calculate the risk loading per policy to see the average capital the insurance company should prepare or charge for one policy to measure the possible risk, and to verify whether diversification will indeed reduce the risk.

We suppose the portfolio of an insurance company contains N policies of a given risk, and assume that each policy will be exposed to n times to the given risk. Therefore the risk will occur n×N times at most. In order to model the occurrence of the risk,we introduce a sequence (Xi, i = 1,...,Nn) of random variable Xi. Each random variable represents the risk occurs or not for one time. Then we set when the risk occurs, the loss, 𝑙, will be 10. Hence, the total loss of portfolio, L, can be expressed as follow:

                                                                     𝐿=𝑙𝑆_{𝑁𝑛}
 

                                                                  𝑆_{𝑁𝑛}=∑𝑖=1-𝑁𝑛 𝑋𝑖
                                                                  
We assume that Xi is iid random variable and is subjected to Bernoulli distribution, 𝐵(𝑝). Hence the total occurence of risk 𝑆_{𝑁𝑛} is subjected to binomial distribution, 𝐵(𝑁𝑛,𝑝). The risk loading per policy R can be expressed as follow:   
   
                                                                    𝑅=𝜂[𝜌(𝐿)𝑁−𝑙𝑛𝑝]
                                                                    
N is the number of policies contained in the portfolio.  𝜌(𝐿)  is the risk measure which we choose VaR and TVaR for this case. We set the times of exposure to risk of one policy n to be 6, the severity l to be 10.

In order to get results with robustness, we set the number of policies N to be 1, 5, 10, 100, 1000 and 10000. The probability of risk occurence p to be 1/6 (fair game),  1/4 , and 1/2 (extreme case). The confidence level of risk measure to be 99%. We used Monte-Carlo simulation to obtain the samples of total of amount. Based on the CLT and Large Number Theorem, the simulation will be executed 100,000 times to realize the convergence. The codes and results are shown below.
'''

# import the packages and libraies 
import numpy as np
import pandas as pd
from scipy.stats import binom


# Basic pre-setting parameters
N = [1,5,10,50,100,1000,10000]
# This P is the probability of risk occourence, 
# which is different from the P in Second Part and Third Part
P = [1/6, 1/4, 1/2] 
l = 10
R_VaR = {'Risk measure': 'VaR','Number N of policies':N}
R_TVaR = {'Risk measure': 'TVaR','Number N of policies':N}

#Simulation
for p in P: # Keep the probability of risk occurence constant
    R_var = []
    R_tvar = []
    for n in N: # Keep the number of policies constant
        expected_loss = l*6*p
        sample_loss = l*np.random.binomial(6*n, p, 1000000) # Simulate the loss 100,0000 times
        var = np.quantile(sample_loss, 0.99) # Caculate the VaR of 99% confidence level
        r_var = (0.15 * (var / n - expected_loss)).round(3) # Calculate risk loading per policy measuered by VaR
        sample_tvar = sample_loss[sample_loss >= var] # Calculate the losses larger than VaR
        R_var.append(r_var)
        tvar = sample_tvar.mean() #Calculate the TVaR
        r_tvar = (0.15 * (tvar / n - expected_loss)).round(3) # Calculate risk loading per policy measured by TVaR
        R_tvar.append(r_tvar)
    R_VaR['p = 1/{}'.format(int(1/p))] = R_var
    R_TVaR['p = 1/{}'.format(int(1/p))] = R_tvar
R_VaR = pd.DataFrame(R_VaR) # Generate the result tables
R_TVaR = pd.DataFrame(R_TVaR)

print(R_VaR)
print('\n')
print(R_TVaR)

'''
From the two tables above, we found that the values of risk loading per policy are very similar to the those on the slides, which means that the performance of our simulation is very well. Comparing the values horizontally, the risk loading per policy increases with the increase of the probability of risk occurence. Hence under the larger probability of risk occurence, the insurance company needs to prepare more capital for the exceedance of expected the loss or charge more on the clients. Comparing the values vertifcally, we found that the risk loading per policy decreases with the increase of number of policies, and the risk loading per policy is nearly 0, if the number of policies is large enough. It means that diversification will decrease the total loss of entire portfolio effectively. Comparing the values of VaR and TVaR, we found that risk loading per policy measured by TVaR is larger than measured by VaR. The reason because the difference of risk measure, and TVaR is larger than VaR basicly.   
'''

'''
Risk Exposure with Different Systematic Risk

In case a, we discussed the relationship between risk loading per policy and both probability of risk occurence and number of policies in the portfolio. We found that risk of entire portfolio can indeed be reduced by diversification. But this conclusion is based on that there is no systematic risk and the probability of risk occurence keeps constant. If the probability of risk occurence changes, the conclution in the first part may be not robust. So in this part, we will relax the assumption, and introduce the systematic risk to check whether diversification is still effective enough when the systematic risk happens.

In order to model the systematic risk, we introduce a random variable U, which is subjected to Bernoulli distribution, B(𝑝̃). We also set the probabiliy of risk occurence as q and p under systematic risk and no systematic risk respectively, and we let q equals to 1/2  and p equals to 1/6 , which q is larger enough than p to represent the systematic risk will have more intense impact than no systematic risk. Hence, the total occurence of risk  𝑆_{𝑁𝑛}, which is conditionally independent random variable, can be expressed as follow:

                                                                 𝑆𝑞~:=𝑆_{𝑁𝑛}|(𝑈=1)∼𝐵(𝑁𝑛,𝑞)
 
                                                                 𝑆𝑝~:=𝑆_{𝑁𝑛}|(𝑈=0)∼𝐵(𝑁𝑛,𝑝)
                                                                 
The mass probability distribution 𝑓_{𝑆} of the total amout of losses 𝑆_{𝑁𝑛} appears as a mixture of the mass probability distributions of  𝑓_{𝑆𝑞~} and 𝑓_{𝑆𝑝~}:

                                                                   𝑓_{𝑆}=𝑝̃𝑓_{𝑆𝑞~}+(1−𝑝̃)𝑓_{𝑆𝑝~}
                                                                   
We also set the probability of systematic risk 𝑝̃ equals to 0, 0.1%, 1.0%, 5%, 10% respectively to see the impacts of different probabilities of systematic risks. Other settings keep the same as case a. The simulation will be executed 100,000 times as well. The codes and results are shown below.                               
'''

N = [1,5,10,50,100,1000,10000]
P = [0,0.001,0.01,0.05,0.1]
R_VaR = {'Risk measure': 'VaR','Number N of policies':N}
R_TVaR = {'Risk measure': 'TVaR','Number N of policies':N}
for p_crisis in P:
    R_var=[]
    R_tvar = []
    for n in N:
        L = np.array([])
        for i in range(0,100000): #Simulate 100,000 times
            U = np.random.binomial(1, p_crisis) # Simulate the crisis state (systemastic risk occurence)
            if U == 0: # Determine whether the systemastic risk occurs
                l = 10 * np.random.binomial(6*n,1/6)              
                L= np.append(L,l)
            else:
                l = 10*np.random.binomial(6*n,1/2)                
                L = np.append(L,l)
        var = np.quantile(L,0.99) # Calculate the VaR of 99% confidence level
        r_var = 0.15*(var/n - (p_crisis*10*3+(1-p_crisis)*10*1)) # Calculate the risk loading per policy measured by VaR
        r_var = r_var.round(3)
        tvar_sample = L[L >= var]
        tvar = tvar_sample.mean()
        r_tvar = 0.15*(tvar/n - (p_crisis*10*3+(1-p_crisis)*10*1)) # Calculate the risk loading per policy measured by TVaR
        r_tvar = r_tvar.round(3)
        R_var.append(r_var)
        R_tvar.append(r_tvar)
    R_VaR['p = {}%'.format(p_crisis*100)] = R_var
    R_TVaR['p = {}%'.format(p_crisis*100)] = R_tvar
R_VaR = pd.DataFrame(R_VaR) # Generate the result tables
R_TVaR = pd.DataFrame(R_TVaR)
print(R_VaR)
print('\n')
print(R_TVaR)

'''
The results show that most values are of risk loading per policy similar to the values on slides. But some values of 1% probabilities of systemastic risk are different with the slides are differnet. That's because the simulation should be executed more than 10 million times to get the convergence results, but we only simulate 100,000 times due to the speed of executing codes and the computating power of our computers. Meanwhile, it's also becasue of the instability of Value at Risk. But we can still find some basic principles from our results.

We found that larger probability of systemastic risk will result in the larger risk loading per policy, meaning that they are positively correlated. The risk loading per policy of systemastic risk is significantly larger than no systemastic risk. However, the risk loading per policy is negatively correlated with the number of policies in the portfolic, which means the diversication will reduce the total risk to some degree. But we also find that the diversification seems not to work if the probability of systemastic risk is relatively large.

So it can be concluded that risk cannot be fully diversified when systemastic risk occurs and the systemastic risk cannot be eliminated. The risk loading per policy measured by VaR and TVaR shows the same results. Our conclusions will not be influenced by the choice of risk measure.
'''

'''
Risk Exposure with Different Scope of Influence of Systemastic Risk

In the last part, we research on the impact of systemastic risk on the entire portfolio. Our assumption is the systemstic risk will influence all policies in the portfolio and all exposures to risk. But the systemastic may not always impact on the all exposures. It may only impact on a few exposures. So we need to relax the assumption futher. In the last part, random varibale U measure the occurence of systematis risk. In order to model that the systemastic risk will only influence a few exposures to risk, we rewrited the random variable U to a squence of i.i.d random varabie(𝑈𝑗, j=1,2,...,n), representing the number j exposure will be influenced by the systematic risk. We rewrited the sequence (Xi, i = 1, 2,..., N), representing the risk occurence of each policy under all exposures to the sequence of random vector (Xj, j = 1, 2,...n), representing the risk occurence of each policy under each exposure. The random vector Xj can be expressed as follow:

                                                                      𝑋𝑗=(𝑋1𝑗,𝑋2𝑗,...,𝑋𝑁𝑗)^𝑇
                                                                      
Hence the total occurence of risk can be expressed as follow:      
 
                                                                         𝑆_{𝑁𝑛}=∑𝑗=1-𝑛 𝑆̃(𝑗)
                                                                         
𝑆̃(𝑗) is the sum of components of Xj, representing the occurence of risk under one exposure to risk. Hence, 𝑆̃(𝑗) folows binomial distribution, but the probability parameter of this distribution depends on the state of systemastic risk:

                                                                        𝑆̃(𝑗)|(𝑈𝑗=1)∼𝐵(𝑁,𝑞)
 

                                                                        𝑆̃(𝑗)|(𝑈𝑗=0)∼𝐵(𝑁,𝑝)

We introduce random varible 𝐴𝑘 to represent how many risk exposures will be influences by systemstic risk. 𝐴𝑘 follows the binomial distrbution, 𝐵(𝑛,𝑝̃). Hence, the probability of total risk occurence of portfolio can be expressed as follow:

                                                 ℙ(𝑆𝑁𝑛=𝑚)=∑𝑘=0-𝑛 𝕡(𝑆𝑁𝑛=𝑚|𝐴𝑘)𝕡(𝐴𝑘)=∑𝑘=0-𝑛 𝐶𝑘𝑛𝑝̃ 𝑘(1−𝑝̃ 𝑛−𝑘ℙ[𝑆̃(𝑘)𝑞+𝑆̃(𝑛−𝑘)𝑝=𝑚]
                                                 
By conditional independence, the total risk occurence with systemastic risk  𝑆̃ (𝑘)𝑞  and the total risk occurence without systemastic risk  𝑆̃ (𝑛−𝑘)𝑝  can be expressed as follow:

                                                               𝑆̃(𝑘)𝑞=∑𝑗=1-𝑘 (𝑆̃(𝑗)|𝑈𝑗=1)∼𝐵(𝑁𝑘,𝑞)
 
                                                           𝑆̃(𝑛−𝑘)𝑝=∑𝑗=1-(𝑛−𝑘) (𝑆̃(𝑗)|𝑈𝑗=0)∼𝐵(𝑁(𝑛−𝑘),𝑝)  

Then the total loss of portfolio  𝐿  can be defined as:

                                                                    𝐿=𝑙(𝑆̃(𝑘)𝑞+𝑆̃(𝑛−𝑘)𝑝)
 
The expected loss of one policy can be defined as:
                                 
                                                               𝐸(𝐿1)=∑𝑘=0-𝑛 ℙ(𝐴𝑘=𝑘)𝑙(𝑘𝑞+(𝑛−𝑘)𝑝)
                                                               
In this case, we expand the number of policies to 100,000. Other settings are the same as the second part. The simulation will be executed 100,000 times. The codes and results are showing as follow.
'''

# Based on the formula above, calculate the expected loss of one policy
def expected_loss(p_crisis, p = 1/6, q=1/2):
    E_1 =  (binom.pmf(0,6, p_crisis)* 10*6*p + binom.pmf(1,6,p_crisis)*10*(1*q +5*p) + binom.pmf(2,6,p_crisis)*10*(2*q+4*p)
            +binom.pmf(3,6,p_crisis)*10*(3*q+3*p))
    E_2 = binom.pmf(4, 6, p_crisis)*10*(4*q+2*p)+binom.pmf(5, 6, p_crisis)*10*(5*q + p)+binom.pmf(6,6,p_crisis)*10*6*q
    return (E_1+E_2).round(2)

# Basic setting
N = [1,5,10,50,100,1000,10000,100000]
P = [0,0.001,0.01,0.05,0.1]
R_VaR = {'Risk measure': 'VaR', 'Number N of policies':N}
R_TVaR = {'Risk measure': 'TVaR', 'Number N of policies':N}
# Simulation
for p_crisis in P:
    R_var=[]
    R_tvar = []
    for n in N:
        L = np.array([])
        for i in range(0,100000): # simulate 100,000 times
            Ak = np.random.binomial(6, p_crisis) # Simulate how many risk exposure will be influcend
            sq = np.random.binomial(n*Ak, 1/2) # Simulate the risk occurence with systemastic risk
            sp = np.random.binomial(n*(6-Ak), 1/6) # Simulate the risk occurence without systemastic risk
            l = 10*(sq+sp) # Obtain the total risk
            L = np.append(L,l)
        var = np.quantile(L,0.99) # Calculate the VaR of 99% confidence level
        r_var = 0.15*(var/n - expected_loss(p_crisis)) # Calculate the risk loading per policy measured by VaR
        r_var = r_var.round(3)
        tvar_sample = L[L >= var]
        tvar = tvar_sample.mean() # Caluculate the TVaR of 99% confidence level
        r_tvar = 0.15*(tvar/n - expected_loss(p_crisis))# Calculate the risk loading per policy measure by TVaR
        r_tvar = r_tvar.round(3)
        R_var.append(r_var)
        R_tvar.append(r_tvar)
    R_VaR['p = {}%'.format(p_crisis*100)] = R_var
    R_TVaR['p = {}%'.format(p_crisis*100)] = R_tvar
R_VaR = pd.DataFrame(R_VaR) # Generate the result tables
R_TVaR = pd.DataFrame(R_TVaR)
print(R_VaR)
print('\n')
print(R_TVaR)

'''
From the results above, only a few values are different from the slides. The same reason as the second part that the simulation did not be simulated 10 million times due to the computing power. But we found the same conclutions as the second part, even if the systemastic risk only influences a few risk exposures, not all.

The risk loading per policy is negatively correlated with the number of policies in the portfolio so that the risk can be reduced by diversification. But the risk loading per policy is positively correlated with the probability of systemstic risk occurence, and when the probability is relatively large, the risk loading per police doesn't converge to 0, even for the portfolio containing more policies. Therefore,that the systemastic risk cannot be eliminated by full diversification is verfied again.

We also find that the values we got in this case are relatively smaller than the values in the second part, which the total loss of entire portfolio will be smaller if the systemastic risk only has a negative impact on a few risk exposures.
'''

'''
Conclusion

Overall, we found some meaningful conclusions through 3 cases. The risk will increase with the increase of probability of risk occurence. The risk loading per policy or risk of the portfolio will be different if the choice of risk measure is different. Basicly, the risk measured by VaR will be smalled than by TVaR or expected shortfall. If the systemastic risk does not exist, the total risk of portfolio will converge to zero with the number of policies increase and can be eliminated by fully diversification. If there is systemastic risk, the total risk of the portfolio will increase with the probability of systemastic risk occurence increases, the effect of diversification will be discounted, and the systemastic risk cannot be eliminated by full diversification. If the systemastic only influences a few risk exposures, the total risk will be smaller.

Hence, if the systemastic risk lasts for a long time and influences the whole economy, it will result in the extremely high risk, large loss, and devastating damage on economy. It's very important for governments, regulations and companies to evaluate, prevent and avoid systemastic risk.
'''



    
