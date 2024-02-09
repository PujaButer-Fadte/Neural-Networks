from data import Data
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(5)
np.random.seed(5)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument("-a","--a",")
    args = parser.parse_args()
    return args


def minover(dataset, n_Max):
    student = np.zeros(dataset.N)
    student_Updated = np.zeros(dataset.N)
    lowest_Stability = float('inf')
    reached_Stability = 0
    counter = 0
    #converged = False
    while True:
        if counter >= n_Max * dataset.P:
            break
        elif counter > 0:
            student = student_Updated
        K_s = []
        for example in range(dataset.P):
            K_s.append(np.dot(student, dataset.vectors[0][example]) * dataset.targets[example] )
        #K_s = np.dot(student.T, dataset.vectors[0]) * dataset.targets
        least_Stable = np.argsort(K_s)[0]
        lowest_Stability_Local = np.min(K_s)        
        student += dataset.vectors[0][least_Stable] * dataset.targets[least_Stable] / dataset.N
        counter += 1
        if lowest_Stability_Local < lowest_Stability:
            lowest_Stability = lowest_Stability_Local
            reached_Stability = 0
        else:
            reached_Stability += 1
            if reached_Stability == dataset.P:
                break
        student_Updated = student
    return student

def generalizationError(student, teacher):
    error = 1/np.pi * np.arccos((np.dot(student, teacher)) / (np.linalg.norm(student) * np.linalg.norm(teacher) ))
    return error

if __name__ == "__main__":
    args = create_arg_parser()
    n_D = 50
    n_Max = 100
    alphas = [0.1, 0.2, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    N = [20, 40]
    E = []

    for N_ in N:
        E_N = []
        for alpha in alphas:
            average_Error = 0
            P = int(alpha * N_)
            for d in range(n_D):
                dataset = Data(P, N_)
                dataset.generateVectors()
                teacher = dataset.generateTargetsbyTeacher()
                #print(dataset.vectors[0])
                student = minover(dataset, n_Max)
                error = generalizationError(student, teacher) 
                average_Error += error
            E_N.append(average_Error / n_D)
        E.append(E_N)

    print(E)

    plt.figure()
    for i, N_ in enumerate(N):
        plt.plot(alphas, E[i], label=f"N = {N_}")
    plt.xlabel('Alpha')
    plt.ylabel('Generalization Error')
    plt.xlim(0, 8.5)
    plt.ylim(0, 0.7)
    plt.title(f'Learning curve as change in generalization error wrt alpha with n_d={n_D} n_max={n_Max}')
    plt.legend()
    plt.show()
    
 



