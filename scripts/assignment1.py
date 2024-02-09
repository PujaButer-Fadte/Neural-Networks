from data import Data
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random

# Make the plots as reproducible as possible
random.seed(5)
np.random.seed(5) 

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sys",
        "--system_sizes",
        action="store_true",
        help="Bonus point 1: Observe the behavior of Ql.s.(alpha) for different system sizes N",
    )
    args = parser.parse_args()
    return args

def perceptronTraining(dataset, max_epochs):
    converged = False
    weight_Vector = np.zeros(dataset.N) #tabula rasa initialization
    weight_Updated = np.zeros(dataset.N) 
    counter = 0
    while counter < max_epochs:
        E = []
        if counter > 0:
            weight_Vector = weight_Updated
        for example in range(dataset.P):
            #print(weight_Vector)
            #print(dataset.vectors[example])
            E_example = np.dot(weight_Vector, dataset.vectors[0][example]) * dataset.targets[example]
            E.append(E_example)
            if E_example <= 0:
                weight_Vector += dataset.vectors[0][example] * dataset.targets[example] / dataset.P
            #print("update")
        if min(E) > 0:
            converged = True
            break
        else:
            weight_Updated = weight_Vector
            counter += 1

    return converged

if __name__ == "__main__":
    args = create_arg_parser()
    n_D = 50
    n_Max = 100
    N = []
    alphas = []
    if args.system_sizes:
        N = [100, 500, 1000]
        #alphas = np.arange(1.5, 2.5, 0.125)        
    else:
        N = [20, 40]
        #alphas = np.arange(0.75, 3.0, 0.25)
    alphas = np.arange(0.75, 3.0, 0.25)
    Q = []
    
    for N_ in N:
        Q_N = []        
        for alpha in alphas:
            n_Converges = 0
            P = int(alpha * N_)
            for d in range(n_D):
                dataset = Data(P, N_)
                dataset.generateVectors()
                dataset.generateTargets()
                  
                if perceptronTraining(dataset, n_Max) == True:
                    n_Converges += 1
            Q_N.append(n_Converges / n_D) 
        Q.append(Q_N)

    #print(Q)

    plt.figure()
    for i, N_ in enumerate(N):
        plt.plot(alphas, Q[i], label=f"N = {N_}")
    plt.xlabel('Alpha')
    plt.ylabel('Q l. s.')
    plt.xlim(0, 4.0)
    plt.ylim(0, 1.1)
    plt.title(f'Plotting Success rate of convergence wrt alpha with n_d={n_D} n_max={n_Max}')
    plt.legend()
    plt.show()

        



        





