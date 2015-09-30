import random
import numpy as np
from itertools import combinations

class Interactor(object):

    def __init__(self, X, y, database, interactor=lambda x, y: y):
        self.X = X
        self.y = y
        self.N, self.D = self.X.shape
        self.database = database
        self.idx = np.arange(self.N)
        self.interactor = interactor
        self.interactions = map(self.convert_interaction, self.database.get_interactions())

    def generate_constraints(self):
        return combinations(self.idx, 3)

    def convert_interaction(self, interaction):
        a, b, c, oou = interaction.a, interaction.b, interaction.c, interaction.oou
        if oou == 0:
            return (b, c, a)
        if oou == 1:
            return (a, c, b)
        if oou == 2:
            return (a, b, c)

    def get_interaction(self):
        constraints = list(self.constraints)
        constraint = random.choice(constraints)
        return {
            i: self.interactor(self.X[i], self.y[i]) for i in constraint
        }

    @property
    def constraints(self):
        return self.possible_constraints - self.existing_constraints

    def add_interaction(self, a, b, c, oou):
        self.database.add_interaction((a, b, c, oou))
        self.existing_constraints |= {(a, b, c)}
