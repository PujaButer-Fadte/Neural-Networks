from data import Data
import numpy as np
import matplotlib.pyplot as plt

def generateRandomDataSets(nD):
    index = 1
    while index <= nD:
        d = Data(P,n)
        d.generateVectors()
        d.generateTargets()
        if (perceptronTraining(d) == True):
            print("solution found for dataset" , index, d.vectors)
        index += 1

def perceptronTraining(d):
    nMax = 100 # maximum number of epochs
    n_epochs = nMax    # n is atmost nMax
    
    weight = np.empty([n_epochs+1,d.N])
    e_term = 0
    e_terms_per_epoch = []
    
    for i in range(0,n_epochs): #epochs
        for timeSweep in range(0,d.P):
            #weight = np.zeros([n_epochs+1,d.N])
            if timeSweep == 0:
                weights_final = np.zeros([d.P+1, d.N])    
                #for j in range(d.P): #samples
                #e_term  = np.dot(weight[i] ,d.vectors[0][j]) * d.targets[j]
                #e_terms_per_epoch.append(e_term)
            e_term = np.dot(weights_final[timeSweep] ,d.vectors[0][timeSweep]) * d.targets[timeSweep]
            e_terms_per_epoch.append(e_term)
            if e_term <= 0:
                #if((i+1) < n_epochs+1):
                #weight[i+1] = weight[i] + 1/d.N * (d.vectors[0][j] * d.targets[j])
                weights_final[timeSweep+1] = weights_final[timeSweep] + 1/d.N * (d.vectors[0][timeSweep] * d.targets[timeSweep])
            else:
                #if((i+1) < n_epochs+1):
                #weight[i+1] = weight[i]
                weights_final[timeSweep+1] = weights_final[timeSweep]
        if min(e_terms_per_epoch) >= 0:
            print(timeSweep, "", i)
            return True 
        else:
            e_terms_per_epoch.clear()
    weights_final = []
    
def plotGraph(P, n):
    c = 0
    if(P <= n):
        c = 1
    elif (P > n):
        k = P - 1
        for i in range(n):
            m = k - i
            fact = np.math.factorial(k) / ((np.math.factorial(i)) * (np.math.factorial(m)))
            c += fact
        exp = 2 ** (1-P)
        c *= exp
    return(c) 

N = [20]
nD = 50
pls = []
for n in N:
    alpha = [0.75,1.0,1.25,1.50,1.75,2.0,2.25,2.50,2.75,3.0]
    for a in alpha:
        P = int(a * n)
        generateRandomDataSets(nD)
        pls.append(plotGraph(P, n))
    plt.step(alpha,pls, label = ('N', n))
    pls.clear()
plt.xlabel("P/N")
plt.ylabel("P l.s.")
plt.legend()
plt.xlim(0,4.0)
plt.ylim(0,1)   
plt.title("Probability l.s.")
plt.show()




