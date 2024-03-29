# P samples of N dimensional vectors
# binary labels -1 or +1
#vectors mean 0 variance 1
import numpy as np

class Data:
    def __init__ (self,P,N):
        self.P = P
        self.N = N
        self.mean = 0
        self.variance = 1
        self.vectors = []
        self.classes = [-1,1]
        self.targets = []  

    def generateVectors(self):
        #self.vectors= []
        self.vectors.append(self.variance * np.random.randn(self.P,self.N)+self.mean)
        #self.vectors = np.concatenate(self.variance * np.random.randn(self.P,self.N)+self.mean)
        #print(self.vectors)

    def generateTargets(self):
        #self.targets = []
        self.targets = np.random.choice(self.classes, size=self.P)
        #print(self.targets)

    def generateTargetsbyTeacher(self):
        teacher = np.random.randn(self.N)
        teacher = teacher / np.linalg.norm(teacher) * np.sqrt(self.N)
        targets = []
        #print(teacher)
        for example in self.vectors[0]:
            targets.append(np.sign(np.dot(teacher, example)))
        self.targets = targets
        return teacher













    
        

