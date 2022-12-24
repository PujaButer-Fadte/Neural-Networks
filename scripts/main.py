from data import Data
import numpy as np
import matplotlib.pyplot as plt

def generateRandomDataSets(nD):
    while nD > 0:
        d.generateVectors()
        d.generateTargets() 
        perceptronTraining()
        print(nD, "number of dataset")
        nD -= 1

def perceptronTraining():
    nMax = 100 # maximum number of epochs
    n = 100    # n is atmost nMax
    
    weight = np.empty([n,d.N])
    e_term = 0

    for i in range(n): #epochs
        if i == 0:
            weight = np.zeros([n,d.N])
        for j in range(d.P): #samples
            e_term  = np.dot(weight[i] ,d.vectors[0][j] * d.targets[j])
            if e_term <= 0:
                if((i+1) < n):
                    weight[i+1] = weight[i] + 1/d.N * (d.vectors[0][j] * d.targets[j])
            else:
                if((i+1) < n):
                    weight[i+1] = weight[i]
    print(weight)    

def plotGraph():
    nFeatures = [5,20,100]
    alpha = 0.75
    alpha_values = [0.75,1.0,1.25,1.50,1.75,2.0,2.25,2.50,2.75,3.0]
    for n in nFeatures:
        pls = []
        alpha = 0.75
        while (alpha <= 3.0):
            P = int(alpha * n)
            c = 0
            if(P <= n):
                c = 1
            elif (P > n):
                for i in range(n):
                    fact = np.math.factorial(P-1) / ((np.math.factorial(i)) * (np.math.factorial(P-1-i)))
                    c += fact
                c *= 2
                c /= np.power(2,P)
            pls.append(float(c)) 
            alpha += 0.25
        print(len(pls))
        print(pls)
        plt.plot(alpha_values,pls)
        plt.xlim(0,4.0)
        plt.ylim(0,1.5)
    
    plt.show()

N = 5
alpha = 0.75
P = int(alpha * N)
nD = 50
d = Data(P,N)
generateRandomDataSets(nD)
plotGraph()





