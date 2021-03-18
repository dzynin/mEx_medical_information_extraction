import werkzeug

werkzeug.cached_property = werkzeug.utils.cached_property
from flask import Flask, request
from flask_restplus import Api, Resource, fields
from flair.data import Sentence, Token
from flair.models import SequenceTagger


NERTagger: SequenceTagger = SequenceTagger.load(model="../Resources/named_entity_recognition_mex_model(custom_flair_embeddings).pt")
POSTagger: SequenceTagger = SequenceTagger.load(model="../Resources/part_of_speech_tagger_mex_model(Def_Word_Flair).pt")


flask_app = Flask(__name__)
app = Api(app=flask_app,
          version="1.0",
          title="mEx Medical SequenceTagger API (NER & POS)",
          description="This API offers an interface to use NER & POS models for text prediction",
          contact="Ammer Ayach",
          contact_email="amay01@dfki.de")

pos_space = app.namespace('pos', description='POS-Tagger API')
ner_space = app.namespace('ner', description='NER-Tagger API')


@pos_space.route("/")
class POS(Resource):
    def get(self):
        return {
            "status": "Got new data"
        }

    def post(self):
        return {
            "status": "Posted new data"
        }


@ner_space.route("/")
class NER(Resource):
    def get(self):
        return {
            "status": "Got new data"
        }

    def post(self):
        return {
            "status": "Posted new data"
        }


name_space = app.namespace('names', description='Manage names')

model = app.model('Name Model',
                  {'name': fields.String(required=True,
                                         description="Name of the person",
                                         help="Name cannot be blank.")})

list_of_names = {}


@name_space.route("/<int:id>")
class MainClass(Resource):

    @app.doc(responses={200: 'OK', 400: 'Invalid Argument', 500: 'Mapping Key Error'},
             params={'id': 'Specify the Id associated with the person'})
    def get(self, id):
        try:
            name = list_of_names[id]
            return {
                "status": "Person retrieved",
                "name": list_of_names[id]
            }
        except KeyError as e:
            name_space.abort(500, e.__doc__, status="Could not retrieve information", statusCode="500")
        except Exception as e:
            name_space.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")

    @app.doc(responses={200: 'OK', 400: 'Invalid Argument', 500: 'Mapping Key Error'},
             params={'id': 'Specify the Id associated with the person'})
    @app.expect(model)
    def post(self, id):
        try:
            list_of_names[id] = request.json['name']
            return {
                "status": "New person added",
                "name": list_of_names[id]
            }
        except KeyError as e:
            name_space.abort(500, e.__doc__, status="Could not save information", statusCode="500")
        except Exception as e:
            name_space.abort(400, e.__doc__, status="Could not save information", statusCode="400")
