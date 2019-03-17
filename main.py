from flask import Flask, redirect, render_template, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('homepage.html')

@app.route('/make_it_new', methods=['GET', 'POST'])
def make_it_new():
    text = request.form['text']
    return render_template('homepage.html', text=text)

@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500
