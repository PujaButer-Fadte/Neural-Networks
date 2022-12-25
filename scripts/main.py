from data import Data
import numpy as np
import matplotlib.pyplot as plt

def generateRandomDataSets(nD):
    index = 0
    while index <= nD:
        d.generateVectors()
        d.generateTargets() 
        if (perceptronTraining() == True):
            print("solution found for dataset" , index)
        index += 1

def perceptronTraining():
    nMax = 100 # maximum number of epochs
    n = nMax    # n is atmost nMax
    
    weight = np.empty([n,d.N])
    e_term = 0
    e_terms_per_epoch = []

    for i in range(n): #epochs
        if i == 0:
            weight = np.zeros([n,d.N])
        for j in range(d.P): #samples
            e_term  = np.dot(weight[i] ,d.vectors[0][j] * d.targets[j])
            e_terms_per_epoch.append(e_term)
            if e_term <= 0:
                if((i+1) < n):
                    weight[i+1] = weight[i] + 1/d.N * (d.vectors[0][j] * d.targets[j])
            else:
                if((i+1) < n):
                    weight[i+1] = weight[i]
        if min(e_terms_per_epoch) > 0:
            return True 
        else:
            e_terms_per_epoch.clear()

def plotGraph():
    nFeatures = [5, 20, 100]
    alpha = 0.75
    alpha_values = [0.75,1.0,1.25,1.50,1.75,2.0,2.25,2.50,2.75,3.0]
    for n in nFeatures:
        pls = []
        alpha = 0.75
        for a in alpha_values:
            P = int(a * n)
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
            pls.append(c) 
            alpha += 0.25
        plt.step(alpha_values,pls, label = ('N', n))
        plt.xlabel("P/N")
        plt.ylabel("P l.s.")
        plt.legend()
        plt.xlim(0,4.0)
        plt.ylim(0,1)   
        plt.title("Probability l.s.")
    plt.show()

N = 5
alpha = 0.75
P = int(alpha * N)
print(P)
nD = 50
d = Data(P,N)
generateRandomDataSets(nD)
plotGraph()





