# The aim of this project is to record the algorithm to simulate Gaussian Copula, Student-t Copula and Clyton Copula.

'''
1. Gaussian Copula

Algorithm of simulation of multivariate normal distribution:

(1) Generate the mean matrix π and coviariance matrix Ξ£ 
(2) Perform a Cholesky decomposition of Ξ£ to obtain the Cholesky factor  Ξ£^(1/2) 
(3) Generate a vector π=(π1,...,ππ)β² of independent standard normal variates
(4) Set π=π+Ξ£^(1/2)π 

X is subjected to multivariate normal distribution ππ(π,Ξ£)

Algorithm of simulation of Gaussian Coupula
(1) Generate πβΌππ(π,Ξ£) by Algorithm of multivariate normal distribution
(2) Return  π=(Ξ¦(π1)...Ξ¦(ππ)) , Ξ¦(ππ) is the distribution function of  ππ 

U is subjected to distribution of Gaussian Coupula
'''
# Import packages
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as st
%config InlineBackend.figure_format = 'retina'
%matplotlib inline

# Simulation of Bivariate Gaussian Copula with both standard Normal Margin Distributions
#Simulating bivariate Gaussian copula X~Nd(0, Ξ)
#The correlation is set to be 0.95
#Fisrt we simulated both marginal distributions Xi is subjected to N(0,1)

u = 0
sigma1 = 1
sigma2 = 1
rho = 0.95
np.random.seed(2)

#Generate the coviriance matrix
cov1 = np.array([[sigma1 ** 2,rho * sigma1 * sigma2],
                [rho * sigma1 * sigma2, sigma2 ** 2]])

#Generate the Cholesky factor by Cholesky decomposition
cho1 = np.linalg.cholesky(cov1)

#Generate a vetor Z = (Z1, Z2) of independent standard normal variates
#Each component we simulate 100,000 times.
Z1 = np.random.normal(0, 1, 10000)
Z2 = np.random.normal(0, 1, 10000)

#Generate a vetor X = (X1, X2)'' which is subject to the multivariate distribution Nd(0, Ξ)
X = np.array([[0], [0]]) + np.dot(cho1, np.array([Z1,Z2]))

X1 = X[0,:]
X2 = X[1,:]

#Plot the bivariate normal distribution, and set the lines of 99.9% quantile
colors1 = '#000000'
area = np.pi * 1**1
plt.scatter(X1,X2,s=area, c=colors1, alpha=0.2)
plt.axvline(np.quantile(X1, 0.999),linestyle='--')
plt.axhline(np.quantile(X2, 0.999),linestyle='')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Bivariate Normal Distribution with Standard Normal Margin distribution')
plt.show()

#Obtain U = (U1, U2)''
U1 = st.norm.cdf(X1)
U2 = st.norm.cdf(X2)

#Plot the Gaussian Copula
colors1 = '#00CED1'
area = np.pi * 1**1
plt.scatter(U1, U2, s=area, c=colors1, alpha=0.2)
plt.xlabel('U1')
plt.ylabel('U2')
plt.title('Gaussian Copula')

plt.show()

# Simulation of Bivariate Gaussian Copula with different standard Normal Margin Distributions
#secondly, we simulated  marginal distributions Xi is subjected to N(0,1) and N(0,2) respectively
u = 0
sigma1 = 1
sigma2 = 2
rho = 0.95
np.random.seed(2)
#Generate the coviriance matrix
cov1 = np.array([[sigma1 ** 2,rho * sigma1 * sigma2],
                [rho * sigma1 * sigma2, sigma2 ** 2]])

#Generate the Cholesky factor by Cholesky decomposition
cho1 = np.linalg.cholesky(cov1)

#Generate a vetor Z = (Z1, Z2) of independent standard normal variates
#Each component we simulate 100,000 times.
Z1 = np.random.normal(0, 1, 10000)
Z2 = np.random.normal(0, 1, 10000)

#Generate a vetor X = (X1, X2)'' which is subject to the multivariate distribution Nd(0, Ξ)
X = np.array([[0], [0]]) + np.dot(cho1, np.array([Z1,Z2]))
X1 = X[0,:]
X2 = X[1,:]
colors1 = '#000000'
area = np.pi * 1**1
plt.scatter(X1,X2,s=area, c=colors1, alpha=0.2)
plt.axvline(np.quantile(X1, 0.999),linestyle='--')
plt.axhline(np.quantile(X2, 0.999),linestyle='--')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Bivariate Normal Distribution with Different Margin Distributions')
plt.show()

#Generate the components of copula
U1 = st.norm.cdf(X1, 0, 1)
U2 = st.norm.cdf(X2, 0, 2)
colors1 = '#00CED1'
area = np.pi * 1**1
plt.scatter(U1, U2, s=area, c=colors1, alpha=0.2)
plt.xlabel('U1')
plt.ylabel('U2')
plt.title('Gaussian Copula')
plt.show()

'''
2. T-copula

Algorithm of simulation of multivariate student distribution
(1) Generate πβΌππ(π,Ξ£) by Algorithm of multivariate gaussian distribution simulation
(2) Generate independently a positive mixing variable  πβΌπΌπ(1/2π£, 1/2π£), πΌπ(πΌ,π½) is the inverse Gamma distribution with shape parameter πΌ and rate parameter π½, and  π£ is the freedom of the marginal t distribution.
(3) Set  π=π+(βπ)π 

X is subjected to distribution of multivariate student distribution with freedom  π£

Algorithm of simulation of t copulaΒΆ
(1) Generate  πβΌπ‘π(π£,π,Ξ£)  by Algorithm of multivariate student distribution
(2) Return  π=(π‘π£(π1)...π‘π£(ππ)), π‘π£(ππ)  is the marginal t distribution function of  ππ  with freedom  π£ 

U is subjected to distribution of t Coupula

In this part, we tried to simulate student t copula with freedom 3. We first simulated the random vector X which is subjected to biviriate t distribution with freedom 3 by Algorithm 1.3, and then obtained the U which is subjected to the t copula by Algorithm 1.4. We set the coeffecients as 0, 0.3, 0.6, 0.9 to see the shape of t copula with different dependences. The margin distributions are all t3 distribution. In order to get a convergent and robust results, the simulation was excuted 10,000 times. 
'''

#Parameters setting
u = 0
sigma1 = 1
sigma2 = 1
Rho = [0,0.3,0.6,0.9]
v = 3
np.random.seed(1)
m = 1



fig = plt.figure(figsize = (20,10))
#Simulation T copulas
for rho in Rho:
    
    #Cholesky Decomposition
    cov1 = np.array([[sigma1 ** 2,rho * sigma1 * sigma2],
                [rho * sigma1 * sigma2, sigma2 ** 2]])

    cho1 = np.linalg.cholesky(cov1)
    X1 = np.zeros(10000)
    X2 = np.zeros(10000)
    
    #Simulation multivariate T distribution
    for i in range(10000):
        z1 = np.random.normal(0, 1)
        z2 = np.random.normal(0, 1)
        Z = np.array([[0], [0]]) + np.dot(cho1, np.array([[z1],[z2]]))
        Z1 = Z[0]
        Z2 = Z[1]
        W = 1 / (np.random.gamma(1.5, 1/1.5))
        W = np.sqrt(W)
        X1[i] = Z1 * W
        X2[i] = Z2 * W
    
    U1 = st.t.cdf(X1,3)
    U2 = st.t.cdf(X2,3)
    
    ax = fig.add_subplot(2,4,m)
    colors1 = '#000000'
    area = np.pi * 1**1
    plt.scatter(X1,X2,s=area, c=colors1, alpha=0.2)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Bivariate T distribution(rho = {})'.format(rho))
   
 
    ax = fig.add_subplot(2,4,m+4)
    colors1 = '#00CED1'
    area = np.pi * 1**1
    plt.scatter(U1, U2, s=area, c=colors1, alpha=0.2)
    ax.set_xlabel('U1')
    ax.set_label('U2')
    ax.set_title('T Copula (rho = {})'.format(rho))
    m += 1

plt.show()

'''
3. Clyton copula

(1) Generate π1, π2 independent standard uniform random variables
(2) Set π2=(1+π1^(βπ)(π2^(-π/(1+π))-1))^(β1/π)

π=(π1,π2)β² is subjected to the Clyaton copula
'''

theta = 1.5

np.random.seed(1)
#Generate U1, V2
U1 = np.random.uniform(0,1,10000)
V2 = np.random.uniform(0,1,10000)
#Generate U2
U2 = np.power((1 + np.power(U1, -theta)*(np.power(V2, -(theta / (1 + theta))) -1 )), -(1/theta))

colors1 = '#00CED1'
area = np.pi * 1**1
plt.scatter(U1, U2, s=area, c=colors1, alpha=0.2)
plt.show()
