import numpy as np
from path import Path
import cPickle as pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from argparse import ArgumentParser

LINKAGE = {
    'Zoo': 0.315845444878,
    'MNIST': 0.44785467128,
    'Fisher Iris': 0.101504761905,
    '20 Newsgroups': 0.615630569242
}
KEY = {
    'ddt': "Vanilla DDT",
    'tr': "Simple",
    'str': "Random",
    'sr': "Smart",
    'var': "Active",
    'hybrid': "Interleaved"
}

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('name')
    argparser.add_argument('dir')
    argparser.add_argument('idx1', type=int)
    argparser.add_argument('idx2', type=int)
    argparser.add_argument('--out_dir', default='out')
    argparser.add_argument('--save', action='store_true')

    return argparser.parse_args()

def get_values(dir, idx):
    values = {}
    for type in ['ddt', 'tr', 'str', 'sr', 'var', 'hybrid']:
        values[type] = {}
        for graph in ['scores', 'costs']:
            values[type][graph] = []
            for i in idx:
                with open(dir / type / '%s-%u.pkl' % (graph, i), 'r') as fp:
                    values[type][graph].append(pickle.load(fp))
    return values

def make_plots(name, out_dir, values):
    fig = plt.figure(figsize=(5.5, 5), dpi=400)
    ax1 = plt.subplot(211)
    ax1.set_ylabel('Data Log Likelihood')
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(-3,3))
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2 = plt.subplot(212)
    ax2.set_ylabel('Triplet Distance from $T^*$')
    ax2.set_xlabel('Iterations')
    for type, vals in values.iteritems():
        costs = np.array(vals['costs']).mean(axis=0)
        scores = np.array(vals['scores']).mean(axis=0)
        plt.subplot(211)
        plt.plot(xrange(0, len(costs) * 20, 20), costs, label=KEY[type], alpha=0.8)
        plt.subplot(212)
        plt.plot(xrange(0, len(scores) * 400, 400), scores, alpha=0.8)
    plt.axhline(LINKAGE[name], ls='dashed', label='Average Linkage')
    plt.subplot(211)
    plt.legend(loc='best', fancybox=True, framealpha=0.5, labelspacing=0.01, prop={'size':10})
    plt.subplot(212)
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    if args.save:
        plt.savefig("%s/%s-result.png" % (out_dir, name), bbox_inches='tight', dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    args = parse_args()

    dir = Path(args.dir)
    name = args.name

    vals = get_values(dir, xrange(args.idx1, args.idx2))
    make_plots(name, args.out_dir, vals)
