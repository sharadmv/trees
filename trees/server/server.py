from webargs import fields
from webargs.flaskparser import use_args
from flask import Flask, render_template, jsonify

class Server(object):

    def __init__(self, interactor, host='0.0.0.0', port=8080):
        self.interactor = interactor
        self.host = host
        self.port = port

        self.app = Flask(__name__, static_url_path='')

    def listen(self):
        self.app.run(host=self.host,
                     port=self.port,
                     debug=True,
                     use_reloader=False)

    def initialize(self):
        self.initialize_static()
        self.initialize_api()

    def initialize_static(self):

        app = self.app
        @app.route('/')
        def index():
            return render_template('index.html')

    def initialize_api(self):

        app = self.app
        @app.route('/api/fetch_interaction')
        def fetch_interaction():
            interaction = self.interactor.sample_interaction()
            result = {i: self.interactor.convert_data(i) for i in interaction}
            return jsonify({
                'status': 'success',
                'interaction': result
            })

        @app.route('/api/add_interaction', methods=["POST"])
        @use_args({
            'a': fields.Int(required=True),
            'b': fields.Int(required=True),
            'c': fields.Int(required=True),
            'oou': fields.Int(required=True),
        })
        def add_interaction(args):
            a, b, c, oou = args['a'], args['b'], args['c'], args['oou']
            if not (a < b < c):
                return jsonify(status='error')
            if not 0 <= oou <= 2:
                return jsonify(status='error')
            self.interactor.add_interaction(a, b, c, oou)
            return jsonify(status='success')
