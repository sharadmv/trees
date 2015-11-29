import random
import numpy as np

class Interactor(object):

    def __init__(self, dataset, database, subset=None):
        self.dataset = dataset
        self.X = dataset.X
        self.y = dataset.y
        self.N, self.D = self.X.shape
        self.database = database
        self.idx = np.array(subset) if subset is not None else np.arange(self.N)
        self.current_interactions = set(map(self.convert_interaction, self.database.get_interactions()))

    def convert_interaction(self, interaction):
        a, b, c, oou = interaction
        if oou == 0:
            return (b, c, a)
        if oou == 1:
            return (a, c, b)
        if oou == 2:
            return (a, b, c)

    def sample_interaction(self):
        a, b, c = random.choice(self.idx), random.choice(self.idx), random.choice(self.idx)

        if ((a, b, c) not in self.current_interactions and
            (a, c, b) not in self.current_interactions and
            (b, c, a) not in self.current_interactions and
            (b, a, c) not in self.current_interactions and
            (c, b, a) not in self.current_interactions and
            (c, a, b) not in self.current_interactions):
            return (a, b, c)
        else:
            return self.sample_interaction()

    def convert_data(self, i):
        return self.dataset.convert(i)

    def add_interaction(self, a, b, c, oou):
        interaction = (a, b, c, oou)
        converted = self.convert_interaction(interaction)
        if converted not in self.current_interactions:
            self.database.add_interaction(interaction)
            self.current_interactions |= {converted}
