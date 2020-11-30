#!flaskNer/bin/python
from flask import Flask, redirect, url_for, request
import spacy
import flair
from flair.data import Sentence, Token
from flair.models import SequenceTagger
import requests
import json


app = Flask(__name__)


tagger = SequenceTagger.load("../Resources/named_entity_recognition_mex_model(custom_flair_embeddings).pt")
print("Done Loading NER-Tagger!!")

@app.route('/')
def index():
    return "Hello, World!"


@app.route('/predict', methods=['POST'])
def predictSentence():
    res = request.get_json()
    sentence = Sentence()
    for i in json.loads(res):
        sentence.add_token(Token(i))

    #print(sentence)

    tagger.predict(sentence)

    #for entity in sentence.get_spans('ner'):
    #    print(entity)
    #    print(entity.text)
    #    print(entity.tag)
    #    print("--------------")
    #print(sentence.to_dict(tag_type='ner'))

    tags = []
    for token in sentence.tokens:
        #print(token.text, token.get_tag('ner').value)
        tags.append(token.get_tag('ner').value)

    res = json.dumps(tags)
    return res


if __name__ == '__main__':
    app.run(debug=True, port=5001)
