from flask import Flask, redirect, render_template, request

import codecs
import json
import os

import numpy as np
from char_rnn_model import *
from train import load_vocab

app = Flask(__name__)

class Args:
    def __init__(self, temperature=0.5, title='Love and Art in the 1920s reads like: ', length=300):
        self.init_dir='all-text/training-output'
        self.temperature=float(temperature)
        self.start_text=' '
        self.title=str(title)
        self.model_path=''
        self.length=int(length)
        self.max_prob=False
        self.seed=-1
        self.evaluate=False


@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/works_cited')
def works_cited():
    return render_template('works-cited.html')

@app.route('/make_it_new', methods=['GET', 'POST'])
def make_it_new():
    temperature = request.form.get('temperature', False)
    title = request.form.get('title', False)
    length = request.form.get('length', False)

    if not temperature:
        temperature=0.5
    if not title:
        title='A New Creation'
    if not length:
        length=100

    args = Args(temperature, title, length)

    with open(os.path.join(args.init_dir, 'result.json'), 'r') as f:
        result = json.load(f)
    params = result['params']

    if args.model_path:
        best_model = args.model_path
    else:
        best_model = result['best_model']

    best_valid_ppl = result['best_valid_ppl']
    if 'encoding' in result:
        args.encoding = result['encoding']
    else:
        args.encoding = 'utf-8'
    args.vocab_file = os.path.join(args.init_dir, 'vocab.json')
    vocab_index_dict, index_vocab_dict, vocab_size = load_vocab(args.vocab_file, args.encoding)

    logging.info('Creating graph')
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('evaluation'):
            test_model = CharRNN(is_training=False, use_batch=False, **params)
            saver = tf.train.Saver(name='checkpoint_saver')

    if args.seed >= 0:
        np.random.seed(args.seed)
    # Sampling a sequence
    with tf.Session(graph=graph) as session:
        saver.restore(session, best_model)
        sample = test_model.sample_seq(session, (5*args.length), args.start_text,
                                        vocab_index_dict, index_vocab_dict,
                                        temperature=args.temperature,
                                        max_prob=args.max_prob)
    return render_template('textpage.html', text=sample, title=title)

@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500
