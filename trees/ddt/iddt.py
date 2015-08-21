from ddt import DirichletDiffusionTree
from inode import InteractiveNonTerminal, InteractiveLeaf

class InteractiveDirichletDiffusionTree(DirichletDiffusionTree):

    def __init__(self, df, likelihood_model, constraints=[]):
        super(InteractiveDirichletDiffusionTree, self).__init__(df, likelihood_model)
        self.constraints = constraints

    def non_terminal(self, *args, **kwargs):
        return InteractiveNonTerminal(*args, **kwargs)

    def leaf(self, *args, **kwargs):
        return InteractiveLeaf(*args, **kwargs)
