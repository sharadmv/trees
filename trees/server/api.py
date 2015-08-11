from webargs import Arg
from webargs.flaskparser import use_args
from server import app
from flask import jsonify
import logging
logging.basicConfig(level=logging.INFO)
from threading import Thread
import seaborn as sns
sns.set_style("white")
import matplotlib.pyplot as plt
import mpld3
import util as server_util

from ..tssb import InteractiveTSSB, TSSB, GibbsSampler, depth_weight, util

class API(object):

    def __init__(self, interactor, parameter_process):
        self.interactor = interactor
        params = {
            'alpha': depth_weight(2, 1),
            'gamma': 0.7,
        }
        self.tssb = TSSB(parameter_process, parameters=params)
        self.constraints = [self.convert_interaction((d.a, d.b, d.c, d.oou)) for d in self.interactor.interactions]
        logging.info("Loaded %u constraints" % len(self.constraints))
        self.itssb = InteractiveTSSB(parameter_process, parameters=params,
                                     constraints=self.constraints)

        self.ll = {
            'TSSB': [],
            'iTSSB': []
        }
        t1 = Thread(target = self.start_tssb, args = (self.tssb, "TSSB"))
        t1.daemon = True
        t1.start()
        t2 = Thread(target = self.start_tssb, args = (self.itssb, "iTSSB"))
        t2.daemon = True
        t2.start()
        self.save_tree = 0
        self.best = {
            'TSSB': (self.tssb, float('-inf')),
            'iTSSB': (self.itssb, float('-inf'))
        }

        @app.route('/api/fetch_interaction')
        def fetch_interaction():
            interaction = self.interactor.get_interaction()
            return jsonify(**{
                'status': 'success',
                'interaction': interaction
            })

        @app.route('/api/get_log_likelihood')
        def get_log_likelihood():
            return jsonify(**{
                'status': 'success',
                'tssb': self.ll['TSSB'],
                'itssb': self.ll['iTSSB'],
            })

        @app.route('/api/save_trees', methods=['POST'])
        def save_trees():
            logging.info("Saving trees...")
            self.save_tree = 2
            return jsonify(**{
                'status': 'success',
            })

        @app.route('/api/get_ll_plot')
        def get_ll_plot():
            fig = plt.figure()
            plt.plot(self.ll['TSSB'], label='tssb')
            plt.plot(self.ll['iTSSB'], label='itssb')
            plt.legend(loc='best')
            result = jsonify(**mpld3.fig_to_dict(fig))
            plt.close('all')
            return result

        @app.route('/api/get_tree_plot')
        @use_args({
            'tree': Arg(str, required=True),
            'dpi': Arg(int, default=80)
        })
        def get_tssb_plot(args):
            fig, ax = plt.subplots(1, 1)
            tssb = self.best[args['tree']][0]
            g, nodes, node_labels = util.plot_tssb(tssb, ax=ax)
            sns.despine()
            labels = []
            for node in g.nodes_iter():
                labels.append(
                server_util.convert_labels_to_html([
                    self.interactor.y[i] for i in node.points
                ]))
            tooltip = mpld3.plugins.PointHTMLTooltip(nodes, labels=labels)
            mpld3.plugins.connect(fig, tooltip)
            tooltip = mpld3.plugins.PointHTMLTooltip(node_labels, labels=labels)
            mpld3.plugins.connect(fig, tooltip)
            plt.axis('off')
            fig.patch.set_visible(False)
            fig.dpi = args['dpi']
            ax.axis('off')
            result = jsonify(**mpld3.fig_to_dict(fig))
            plt.close('all')
            return result

        @app.route('/api/add_interaction', methods=["GET", "POST"])
        @use_args({
            'a': Arg(int, required=True),
            'b': Arg(int, required=True),
            'c': Arg(int, required=True),
            'oou': Arg(int, required=True),
        })
        def add_interaction(args):
            a, b, c, oou = args['a'], args['b'], args['c'], args['oou']
            if not (a < b < c):
                return jsonify(status='error')
            if not 0 <= oou <= 2:
                return jsonify(status='error')
            self.add_interaction(a, b, c, oou)
            return jsonify(status='success')

    def convert_interaction(self, interaction):
        a, b, c, oou = interaction
        if oou == 0:
            return (b, c, a)
        if oou == 1:
            return (a, c, b)
        if oou == 2:
            return (a, b, c)

    def add_interaction(self, a, b, c, oou):
        self.interactor.add_interaction(a, b, c, oou)
        self.itssb.add_constraint(self.convert_interaction((a, b, c, oou)))

    def start_tssb(self, tssb, name):
        gs = GibbsSampler(tssb, tssb.parameter_process, self.interactor.X)
        gs.initialize_assignments()
        i = 0
        while True:
            i += 1
            gs.gibbs_sample()
            ll = tssb.marg_log_likelihood(self.interactor.X)
            self.best[name] = tssb.copy(), ll
            logging.info("%s -- Iteration %u: %f" % (name, i, ll))
            if name == "iTSSB":
                logging.info("%s -- %u constraints" % (name, len(tssb.constraints)))
            self.ll[name].append(tssb.marg_log_likelihood(self.interactor.X))
            if self.save_tree > 0:
                self.save_tree -= 1
                util.save_tssb(tssb, '%s.pkl' % name)
