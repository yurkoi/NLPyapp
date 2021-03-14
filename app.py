from pprint import pprint

from flask import Flask, render_template, request, redirect, send_file, make_response, jsonify, json
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from utils import img_fig, default_statistic, text_blob_polarity, ner_analyzer, \
    sentences_statistic, custom_ner, semantic_similarity
import json
import time
from io import StringIO
import csv
import spacy
nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///nlpy.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = 'static/files/original'
path_to_file = ''
data = []


class Article(db.Model):
    id = db.Column(db.Integer, primary_key=True, nullable=True)
    title = db.Column(db.String(100), primary_key=True)
    intro = db.Column(db.String(300), primary_key=True, nullable=False)
    text = db.Column(db.Text, primary_key=True)
    date = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return "<Article %r>" % self.id


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/semantic-similarity', methods=['GET', 'POST'])
def seman_similarity():
    if request.method == 'POST':
        text_1 = request.form['text_1']
        text_2 = request.form['text_2']
        res = semantic_similarity(text_1=text_1, text_2=text_2)
        # doc1 = nlp(text_1)
        # doc2 = nlp(text_2)

        # res = round(doc1.similarity(doc2), 3)
        return render_template('semantic_similarity.html', cosine_value=res, text_1=text_1, text_2=text_2)

    else:
        return render_template('semantic_similarity.html')


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    start = time.time()
    if request.method == 'POST':

        rawtext = request.form['text']
        data_res = default_statistic(rawtext)
        ner_defaul = ner_analyzer(rawtext)
        plar_subjectivity = text_blob_polarity(rawtext)

        end = time.time()
        final_time = round(end-start, 4)

        return render_template('analyze.html', performance=final_time, length_of_the_doc=data_res['length_of_doc'],
                               alpha_perc=data_res['perc_alpha'], digit_perc=data_res['perc_digit'],
                               stop_w_words=data_res['pers_stop_w'], mean_tok_shape=data_res['mean_tok_shape'],
                               med_tok_shape=data_res['med_tok_shape'], max_tok_shape=data_res['max_tok_stat'],
                               poses=data_res['poses'], depp=data_res['depp'], ner_def=ner_defaul['default_ner'],
                               ner_stats=ner_defaul['pers_ner'],
                               polar_subj=plar_subjectivity,
                               rawtext=rawtext)
    return render_template('analyze.html')


@app.route('/sentence-analyze', methods=['GET', 'POST'])
def sent_analyze():
    start = time.time()
    if request.method == 'POST':
        rawtext = request.form['text']
        result = sentences_statistic(rawtext)

        end = time.time()
        final_time = round(end - start, 4)

        return render_template('sentence_analyzer.html', final_time=final_time, tables=[result.to_html(classes='data')],
                               titles=result.columns.values, raw_text=rawtext)
    else:
        return render_template('sentence_analyzer.html')


@app.route('/posts')
def posts():
    articles = Article.query.order_by(Article.date.desc()).all()
    return render_template('posts.html', articles=articles)


@app.route('/posts/<int:id>')
def post(id):
    article = Article.query.get(id)
    return render_template('post.html', articles=article)


@app.route('/create-article', methods=['POST', 'GET'])
def create_article():
    if request.method == "POST":
        id = int(request.form['id'])
        title = request.form['title']
        intro = request.form['intro']
        text = request.form['text']

        article = Article(id=id, title=title, intro=intro, text=text)
        try:
            db.session.add(article)
            db.session.commit()
            return redirect("/index")
        except:
            return "<h1> Error while article adding </h1>"
    else:
        return render_template('create-article.html')


@app.route('/word-cloud', methods=['POST', 'GET'])
def word_cloud():
    if request.method == "POST":
        raw_text = request.form['text']
        image = img_fig(raw_text)

        return send_file(image, mimetype='image/png')
    else:
        return render_template('word-cloud.html')


@app.route('/custom-ner', methods=['POST', 'GET'])
def custm_ner():
    if request.method == 'POST':
        file = request.files['file']
        raw_text = file.read().decode()

        pattern = request.form['sub_patterns']
        pattern_name = request.form['pattern_name']
        result, raw_text_len, train_text_len = custom_ner(raw_text, pattern, pattern_name)
        global data
        data = result

        return render_template('custom_ner.html', result=result, raw_len=raw_text_len,
                               train_len=train_text_len)
    else:
        return render_template('custom_ner.html', train_len=0)


@app.route('/download')
def get_jsonner():
    # data = make_summary()
    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
    return response


# @app.route('/download', methods=['POST', 'GET'])
# def sent_csv():
#     si = StringIO()
#     cw = csv.writer(si)
#     cw.writerows(data)
#     output = make_response(si.getvalue())
#     output.headers["Content-Disposition"] = "attachment; filename=train_df.csv"
#     output.headers["Content-type"] = "text/csv"
#     return output


if __name__ == "__main__":
    app.run(debug=True)
