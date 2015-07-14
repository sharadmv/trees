import numpy as np
import scipy.stats as stats

from ..distribution import Distribution, LazyGEM

depth_weight = lambda a, l: lambda j: l ** j * a

class TSSB(Distribution):

    def __init__(self, index=(), depth=0, *args, **kwargs):
        super(TSSB, self).__init__(*args, **kwargs)
        self.index = index
        self.depth = depth

        alpha = self.get_parameter("alpha")
        gamma = self.get_parameter("gamma")

        self.nu = stats.beta(1, alpha(self.depth)).rvs()
        self.psi = LazyGEM({
            'a': gamma
        })
        self.children = []
        self.cur_index = -1

    def get_child(self, key):
        alpha = self.get_parameter("alpha")
        gamma = self.get_parameter("gamma")

        if self.cur_index < key:
            while self.cur_index < key:
                self.cur_index += 1
                self.children.append(TSSB(
                    parameters={
                        'alpha': alpha,
                        'gamma': gamma
                    },
                    index=self.index + (self.cur_index,),
                    depth=self.depth + 1,
                ))
        return self.children[key]

    def uniform_index(self, u):
        if u < self.nu:
            return self.index
        u = (u - self.nu) / (1.0 - self.nu)
        i, right_weight = self.psi.uniform_index(u)
        child, weight = self.get_child(i), self.psi[i]
        left_weight = right_weight - weight
        u = (u - left_weight) / (weight)
        return child.uniform_index(u)

    def sample_one(self):
        return self.uniform_index(np.random.random())

    def __repr__(self):
        if self.index:
            return '-'.join(map(str, self.index))
        return '-'

    def parameters(self):
        return {"alpha", "gamma"}
