class Stream(object):

    def __getitem__(self, index):
        raise NotImplementedError

class DistributionStream(Stream):

    def __init__(self, distribution):
        self.dist = distribution
        self.vals = []
        self.cur_index = -1

    def __getitem__(self, index):
        if index > self.cur_index:
            diff = index - self.cur_index
            self.vals.extend(self.dist.sample(diff))
        return self.vals[index]
