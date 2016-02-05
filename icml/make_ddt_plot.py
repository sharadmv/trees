import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from trees.ddt import *
sns.set_style('white')

S = 1

increment = 0.001

times = np.arange(0, 1 + increment, increment).tolist()

def get_time(t):
    return times[int(t / increment)]

class Node(object):
    def __init__(self, path, time, parent, leaf=False):
        self.path = path
        self.parent = parent
        self.leaf = leaf
        self.children = []
        self.time = time

    def sample_point(self):
        r = random.random()
        total_count = float(self.leaf_count())
        for i, c in enumerate(self.children):
            leaf_count = c.leaf_count()
            if r > leaf_count / total_count:
                continue
            diverge = random.random() > df.no_divergence(times[self.time], times[c.time], leaf_count)
            if diverge:
                time = times.index(get_time(df.sample(times[self.time], times[c.time], leaf_count)[0]))
                path = c.get_path()
                first_path = path[:time]
                motion = brownian_motion(first_path[-1], (times[time], 1.0))
                new_path = first_path + motion
                while len(new_path) > len(times):
                    new_path = new_path[1:]
                assert len(new_path) == len(times), (len(new_path), len(times))
                node = Node(None, time, self)
                leaf = Node(new_path, -1, node, True)
                node.children.append(leaf)
                node.children.append(c)
                c.parent = node
                self.children[i] = node
                return leaf
            return c.sample_point()

    def get_path(self):
        if self.leaf:
            return self.path
        else:
            return self.left.get_path()[:self.time + 1]

    def leaf_count(self):
        if self.leaf:
            return 1
        return sum([c.leaf_count() for c in self.children])

    @property
    def location(self):
        return self.get_path()[-1]

    @property
    def left(self):
        return self.children[0]

    @property
    def right(self):
        return self.children[1]

    def __repr__(self):
        if self.leaf:
            return "Node(%f, %f)" % (times[self.time], self.get_path()[-1])
        return "Node(%f, %f, %s)" % (times[self.time], self.get_path()[-1], '(%s)' % ', '.join(str(c) for c in self.children))

def brownian_motion(start, time_interval):
    start_time, end_time = time_interval
    vals = [start]
    for i in np.arange(start_time, end_time, increment):
        vals.append(np.random.normal(loc=vals[-1], scale=np.sqrt(S) * increment))
    return vals

df = Inverse(c = 0.7)

root = Node(None, 0, None)
x1 =  Node(brownian_motion(0, (0, 1)), len(times) - 1, root, leaf=True)
root.children.append(x1)

wat = [x1, root.sample_point(), root.sample_point(), root.sample_point()]

def get_path(point):
    return point.path


def plot_node(node):
    if node.leaf:
        plt.plot(times, node.path, alpha=0.5, color=current_palette[0])
        plt.plot(1.0, node.get_path()[-1], 'o', color='black')
        return
    for c in node.children:
        plot_node(c)
    plt.plot(times[node.time], node.get_path()[-1], 'o', color='black')

current_palette = sns.color_palette()
plot_node(root)
plt.xlabel('Time')
plt.savefig('out/ddt.png', bbox_inches='tight')
