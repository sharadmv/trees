from argparse import ArgumentParser
import server
import api
from interact import Interactor
from database import Database
from api import API

from ..data import load

def parse_args():
    argparser = ArgumentParser()

    argparser.add_argument("--port", "-p", type=int, default=8080)
    argparser.add_argument("--debug", action='store_true')
    argparser.add_argument("--dataset", type=str, default='zoo')
    argparser.add_argument("--host", type=str, default='0.0.0.0')

    return argparser.parse_args()

def main():
    args = parse_args()
    X, y, process = load(args.dataset)
    db = Database('%s.db' % args.dataset)
    interactor = Interactor(X, y, db)
    api = API(interactor, process)
    server.listen(host=args.host, port=args.port, debug=args.debug)
