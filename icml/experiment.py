import cPickle as pickle
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from path import Path

from trees.mcmc import MetropolisHastingsSampler
from trees.ddt import *
from trees.data import load
from trees.constraint import *
from util import plot_tree, plot_tree_2d, dist
from master_tree import make_master

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('--data', default='mnist')
    argparser.add_argument('--subset', default=100, type=int)
    argparser.add_argument('--iters', default=10000, type=int)
    argparser.add_argument('--cont', action='store_true')
    argparser.add_argument('--out_dir', default='out')
    argparser.add_argument('i', type=int)
    return argparser.parse_args()

def run_experiment(index, dataset_name, name, constraint_getter, master_tree, X, y, out_dir, n_iters=1000, add_constraint=200, add_score=200,
                   add_likelihood=200, should_continue=False):


    N, D = X.shape
    df = Inverse(c=1)

    if dataset_name == 'iris':
        lm = GaussianLikelihoodModel(sigma=np.eye(D) / 9.0, sigma0=np.eye(D) / 2.0, mu0=X.mean(axis=0)).compile()
    elif dataset_name == 'zoo':
        lm = GaussianLikelihoodModel(sigma=np.diag(np.diag(np.cov(X.T))) / 4.0, sigma0=np.eye(D) / 2.0, mu0=X.mean(axis=0)).compile()
    else:
        lm = GaussianLikelihoodModel(sigma=np.diag(np.diag(np.cov(X.T))) / 2.0, sigma0=np.eye(D) / 2.0, mu0=X.mean(axis=0)).compile()
    if should_continue:
        with open(out_dir / name / 'scores-%u.pkl' % index, 'r') as fp:
            scores = pickle.load(fp)
        with open(out_dir / name / 'costs-%u.pkl' % index, 'r') as fp:
            costs = pickle.load(fp)
        with open(out_dir / name / 'final-tree-%u.pkl' % index, 'r') as fp:
            tree = DirichletDiffusionTree(df=df, likelihood_model=lm)
            tree.set_state(pickle.load(fp))
        sampler = MetropolisHastingsSampler(tree, X)
    else:
        scores = []
        costs = []
        tree = DirichletDiffusionTree(df=df, likelihood_model=lm)
        sampler = MetropolisHastingsSampler(tree, X)
        sampler.initialize_assignments()
        if dataset_name == 'zoo':
            sampler.tree = sampler.tree.induced_subtree(master_tree.points())

    current_run = []
    for i in tqdm(xrange(n_iters + 1)):
        sampler.sample()
        current_run.append(sampler.tree)
        if i % add_score == 0:
            scores.append(dist(master_tree, sampler.tree))
        if i % add_likelihood == 0:
            costs.append(sampler.tree.marg_log_likelihood())
        if i != 0 and i % add_constraint == 0:
            if constraint_getter is not None:
                constraint = constraint_getter.get_constraint(current_run)
                if constraint is not None:
                    sampler.add_constraint(constraint)
            current_run = []
    # plot_tree(sampler.tree, y)

    (out_dir / name).mkdir_p()
    with open(out_dir / name / 'scores-%u.pkl' % index, 'w') as fp:
        pickle.dump(scores, fp)
    print len(costs)
    with open(out_dir / name / 'costs-%u.pkl' % index, 'w') as fp:
        pickle.dump(costs, fp)
    # with open(out_dir / name / 'trees-%u.pkl' % index, 'r') as fp:
        # previous_trees = pickle.load(fp)
    # with open(out_dir / name / 'trees-%u.pkl' % index, 'w') as fp:
        # pickle.dump(previous_trees + [t.get_state() for t in trees], fp)
    with open(out_dir / name / 'final-tree-%u.pkl' % index, 'w') as fp:
        pickle.dump(sampler.tree.get_state(), fp)
    return costs, scores, sampler

if __name__ == "__main__":
    args = parse_args()

    out_dir = Path(args.out_dir) / args.data
    out_dir.mkdir_p()
    dataset_name = args.data
    dataset = load(dataset_name)
    X, y = dataset.X, dataset.y

    if dataset_name == 'mnist' or dataset_name == 'iris' or dataset_name == '20news':
        np.random.seed(0)
        idx = np.random.permutation(xrange(X.shape[0]))[:args.subset]
        X = X[idx]
        y = y[idx]
    if dataset_name == 'mnist' or dataset_name == '20news':
        pca = PCA(10)
        X = pca.fit_transform(X)
    if dataset_name == 'zoo':
        # pca = PCA(5)
        # X = pca.fit_transform(X)
        X += np.random.normal(size=X.shape) * 0.01

    master_tree = make_master(X, y, dataset_name)

    experiments = [('ddt', None),
                   ('tr', TotallyRandom(master_tree, y, classification=dataset_name != 'zoo')),
                   ('str', StupidRandom(master_tree, y, K=10, classification=dataset_name != 'zoo')),
                   ('sr', SmartRandom(master_tree, y, classification=dataset_name != 'zoo')),
                   ('var', Variance(master_tree, y, N=20, K=10, classification=dataset_name != 'zoo')),
                   ('hybrid', Hybrid(master_tree, y, N=20, K=10, classification=dataset_name != 'zoo'))
                   ]

    # plt.figure()
    overall_cost, overall_score = [], []
    samplers = []
    for name, type in experiments:
        logging.info("Running %s", name)
        costs, scores, sampler = run_experiment(args.i, dataset_name, name, type, master_tree, X, y, out_dir,
                                                add_score=400,
                                                add_constraint=100,
                                                add_likelihood=20, n_iters=args.iters,
                                                should_continue=args.cont)
        overall_cost.append(costs)
        overall_score.append(scores)
        # plt.subplot(211)
        # plt.plot(costs, label=name)
        # plt.subplot(212)
        # plt.plot(scores, label=name)
        samplers.append(sampler)

    # plt.legend(loc='best')
    # plt.show()

    # plt.figure()
    # plot_tree(samplers[-1].tree, y)
    # plt.figure()
    # pca2 = PCA(2)
    # X2 = pca2.fit_transform(X)
    # plot_tree_2d(samplers[-1].tree, X2)
    # plt.show()
