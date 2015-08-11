from flask import Flask, render_template

app = Flask(__name__, static_url_path='')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/trees')
def trees():
    return render_template('trees.html')

def listen(host='0.0.0.0', port=8080, debug=True):
    app.run(debug=debug, host=host, port=port, use_reloader=False)
