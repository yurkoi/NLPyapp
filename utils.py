from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
import pandas as pd
import string
from spacy.matcher import Matcher
from spacy.lang.en import English

import spacy
from textblob import TextBlob
nlp = spacy.load('en_core_web_sm')

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


def img_fig(my_text):
    plt.figure(figsize=(20, 10))
    word_cloud = WordCloud(background_color='white', mode="RGB",
                           width=2000, height=1000).generate(my_text)
    plt.imshow(word_cloud)
    plt.axis("off")
    img = BytesIO()
    plt.savefig(img)
    img.seek(0)
    return img


def hist(data):
    data.hist(sharey=True, layout=(3, 3), figsize=(15, 8))
    img = BytesIO()
    plt.savefig(img)
    img.seek(0)
    return img


def patterns_splitter(raw_shit):
    patterns = [pattern.split(',') for pattern in raw_shit.split('/')]

    all_patterns = []
    for pattern in patterns:
        if len(pattern) > 1:
            long_pattern = {}
            for sub_pattern in pattern:
                s_p = sub_pattern.split(':')
                long_pattern[s_p[0]] = s_p[1]
            all_patterns.append([long_pattern])
        else:
            short_pattern = {}
            p_t = pattern[0].split(':')
            short_pattern[p_t[0]] = p_t[1]
            all_patterns.append([short_pattern])

    return all_patterns


def custom_ner(text, sub_patterns, pattern_name):
    TEXTS = sent_tokenize(text)
    raw_text_len = len(TEXTS)

    nlp = English()
    matcher = Matcher(nlp.vocab)

    sub_patterns = patterns_splitter(sub_patterns)
    matcher.add(pattern_name, None, *sub_patterns)

    training_data = []
    for doc in nlp.pipe(TEXTS):
        spans = [doc[start:end] for match_id, start, end in matcher(doc)]
        entities = [(span.start_char, span.end_char, pattern_name) for span in spans]
        if len(entities) > 0:
            training_example = (doc.text, {"entities": entities})
            training_data.append(training_example)
    # print(training_data)
    training_data_len = len(training_data)

    # print(pd.DataFrame(training_data, columns=['content', 'entities']).to_csv(f'{pattern_name}_frame.csv'))

    return training_data, raw_text_len, training_data_len


def semantic_similarity(text_1, text_2):
    doc1 = nlp(text_1)
    doc2 = nlp(text_2)
    similarity = round(doc1.similarity(doc2), 3)
    return similarity


def ner_analyzer(raw_text):
    doc = nlp(raw_text)
    default_ner = [(entity.text, entity.label_)for entity in doc.ents]
    ners_shape = len(default_ner)
    ners = pd.Series([i[1] for i in default_ner]).value_counts()

    pers_ner = []
    for ner in zip(ners, ners.index):
        s = (ner[1], round(ner[0]/ners_shape, 3))
        pers_ner.append(s)
    return {'default_ner': default_ner, 'pers_ner': pers_ner}


def text_blob_polarity(rawtext):
    blob = TextBlob(rawtext)
    blob_sentiment, blob_subjectivity = round(blob.sentiment.polarity, 3), round(blob.sentiment.subjectivity, 3)
    return blob_sentiment, blob_subjectivity


def default_statistic(raw_text):
    doc = nlp(raw_text)

    doc_shape = len(doc)
    token_pos = []
    token_dep = []
    token_shape = []
    token_is_alpha = []
    token_is_digit = []
    token_is_stop_w = []

    for token in doc:
        # Get the token text, part-of-speech tag and dependency label
        token_pos.append(token.pos_)
        token_dep.append(token.dep_)
        token_shape.append(len(token))
        token_is_alpha.append(token.is_alpha)
        token_is_digit.append(token.is_digit)
        token_is_stop_w.append(token.is_stop)

    res_pos = pd.Series(token_pos).value_counts()
    res_dep = pd.Series(token_dep).value_counts()
    poses, depp = [], []

    for pos in zip(res_pos.index, res_pos):
        s = (spacy.explain(pos[0]), round(pos[1] / doc_shape, 3))
        poses.append(s)

    for dep in zip(res_dep.index, res_dep):
        s = (dep[0], round(dep[1] / doc_shape, 3))
        depp.append(s)

    # stat of shapes
    mean_stat = round(np.mean(token_shape), 3)
    med_stat = round(np.median(token_shape), 3)
    max_stat = np.max(token_shape)

    # percentage
    perc_alpha = round(sum(token_is_alpha) / doc_shape, 3)
    perc_digit = round(sum(token_is_digit) / doc_shape, 3)
    perc_stop_w = round(sum(token_is_stop_w) / doc_shape, 3)

    all_data = {'length_of_doc': doc_shape, 'perc_alpha': perc_alpha, 'perc_digit': perc_digit,
                'pers_stop_w': perc_stop_w, 'mean_tok_shape': mean_stat, 'med_tok_shape': med_stat,
                'max_tok_stat': max_stat, 'poses': poses, 'depp': depp}

    return all_data


def sentences_statistic(raw_text):

    text = sent_tokenize(raw_text.lower())
    data = pd.DataFrame(pd.Series(text), columns=['content'])

    data['word_count'] = data['content'].apply(lambda x: len(str(x).split()))
    data['unique_word_count'] = data['content'].apply(lambda x: len(set(str(x).split())))
    data['stop_word_count'] = data['content'].apply(
        lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
    data['url_count'] = data['content'].apply(
        lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))
    data['mean_word_length'] = data['content'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    data['char_count'] = data['content'].apply(lambda x: len(str(x)))
    data['punctuation_count'] = data['content'].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation]))
    data['hashtag_count'] = data['content'].apply(lambda x: len([c for c in str(x) if c == '#']))
    data['mention_count'] = data['content'].apply(lambda x: len([c for c in str(x) if c == '@']))

    return round(data.describe(), 3)
