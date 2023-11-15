import numpy as np
from munkres import Munkres

class Feature_Matcher():
    def __init__(self):
        self.munkres = Munkres()

    def hungarian_matching(self, cost_matrix, type='min'):
        if type != 'min' and type!='max': raise Exception("Invalid type for algorithm")
        elif type == 'max':
            max = cost_matrix[0][0]
            for r in cost_matrix:
                for el in r:
                    if el > max: max = el
            for i in range(len(cost_matrix)):
                for j in range(len(cost_matrix[0])):
                    cost_matrix[i][j] = max - cost_matrix[i][j]
        return self.munkres.compute(cost_matrix)
    
    def generate_cost_matrix(self, detections1, detections2):
        #calculate similarities between det1 and det2
        return
    
mat = Feature_Matcher()
matrix = [[10,15,9],
          [9,18,5],
          [6,14,3]]
cost = mat.hungarian_matching(matrix, 'max')
print(cost, matrix)
