import werkzeug

werkzeug.cached_property = werkzeug.utils.cached_property

from flask import Flask, request
from flask_restplus import Api, Resource, fields
from flair.data import Sentence
from flair.models import SequenceTagger

NERTagger: SequenceTagger = SequenceTagger.load(model="ner.pt")
POSTagger: SequenceTagger = SequenceTagger.load(model="pos.pt")

flask_app = Flask(__name__)
app = Api(app=flask_app,
          version="1.0",
          title="mEx Medical SequenceTagger API (NER & POS)",
          description="This API offers an interface to use NER & POS models for text prediction",
          contact="Ammer Ayach",
          contact_email="amay01@dfki.de")

pos_space = app.namespace('pos', description='POS-Tagger API')
ner_space = app.namespace('ner', description='NER-Tagger API')

model = app.model('Taggers Input',
                  {'text': fields.String(required=True,
                                         description="Input text to apply POS & NER tagger",
                                         help="Text cannot be blank.")})

model2 = app.model('Taggers Input', {'list': app.as_list(fields.Nested(model))})


@pos_space.route("/")
class POS(Resource):
    @app.doc(responses={200: 'OK', 400: 'Invalid Argument'})
    @app.expect(model2)
    def post(self):
        try:
            json_data = request.json
            text = json_data["list"]
            print(text)

            sentence = Sentence(text=text)
            POSTagger.predict(sentence)

            tags = []
            tokens = []
            for token in sentence.tokens:
                tokens.append(token.text)
                tags.append(token.get_tag('pos').value)

            return {
                "text": text,
                "tokens": tokens,
                "tags": tags,
                "tagged_text": sentence.to_tagged_string()
            }
        except Exception as e:
            pos_space.abort(400, e.__doc__, status="Could not perform prediction", statusCode="400")


@ner_space.route("/<string:text>")
class NER(Resource):

    @app.doc(responses={200: 'OK', 400: 'Invalid Argument'},
             params={'text': 'Input text to apply POS & NER tagger'})
    @app.expect(model)
    def post(self, text):
        try:
            sentence = Sentence(text=text)
            NERTagger.predict(sentence)

            tags = []
            tokens = []
            for token in sentence.tokens:
                tokens.append(token.text)
                tags.append(token.get_tag('ner').value)

            return {
                "text": text,
                "tokens": tokens,
                "tags": tags,
                "tagged_text": sentence.to_tagged_string()
            }
        except Exception as e:
            ner_space.abort(400, e.__doc__, status="Could not perform prediction", statusCode="400")


if __name__ == '__main__':
    flask_app.run()