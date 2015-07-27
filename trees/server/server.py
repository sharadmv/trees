from argparse import ArgumentParser
from flask import Flask

app = Flask(__name__)

def listen(host='0.0.0.0', port=8080, debug=True):
    app.run(debug=debug, host=host, port=port, use_reloader=False)

def parse_args():
    argparser = ArgumentParser()

    argparser.add_argument("--port", "-p", type=int, default=8080)
    argparser.add_argument("--debug", action='store_true')
    argparser.add_argument("--host", type=str, default='0.0.0.0')

    return argparser.parse_args()

def main():
    args = parse_args()
    listen(host=args.host, port=args.port, debug=args.debug)
