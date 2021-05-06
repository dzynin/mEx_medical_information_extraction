import werkzeug

werkzeug.cached_property = werkzeug.utils.cached_property

from flask import Flask
from flask_restplus import Api, Resource, fields, reqparse
from RelationExtraction import RelationExtractionModel

relex = RelationExtractionModel()

flask_app = Flask(__name__)
app = Api(app=flask_app,
          version="1.0",
          title="mEx Medical SequenceTagger API (Relation Extraction)",
          description="This API offers an interface to use RelEx models for text prediction",
          contact="Ammer Ayach",
          contact_email="amay01@dfki.de")

relex_space = app.namespace('relex', description='RelEx API')

model = app.model('RelEx Input',
                  {'token_list': fields.List(fields.String(required=True,
                                                           description="List of comma separated tokens (str) which form a text",
                                                           help="List cannot be blank.")),
                   'sent_starts': fields.List(fields.String(required=True,
                                                            description="List of comma separated int indices to define where each sentence starts",
                                                            help="List cannot be blank.")),
                   'ner_tags': fields.List(fields.String(required=True,
                                                         description="List of comma separated ner tags (str) same lenght of the token list maps tags to tokens",
                                                         help="Follow BIO format: O-No Tag, B-Tag starts here, I-Tag span"))})

parser = reqparse.RequestParser()
parser.add_argument('token_list', type=list, action='append', required=True, help="List cannot be blank.")
parser.add_argument('sent_starts', type=list, action='append', required=True, help="List cannot be blank.")
parser.add_argument('ner_tags', type=list, action='append', required=True,
                    help="Follow BIO format: O-No Tag, B-Tag starts here, I-Tag span")


# /<token_list>/<sent_starts>/<ner_tags>
@relex_space.route("/")
@app.doc(responses={200: 'OK', 400: 'Invalid Argument'})
class RELEx(Resource):
    # @app.doc(responses={200: 'OK', 400: 'Invalid Argument'},
    #         params={'token_list': 'List of comma separated tokens (str) which form a text',
    #                 'sent_starts': 'List of comma separated int indices to define where each sentence starts',
    #                 'ner_tags': 'List of comma separated ner tags (str) same lenght of the token list maps tags to tokens'}, parser=parser)
    @app.doc(parser=parser)
    @app.expect(model)
    def post(self):
        args = parser.parse_args()
        # print(args)
        token_list = list(map(lambda x: ''.join(x), args['token_list']))
        # token_list = [item for sublist in args['token_list'] for item in sublist]
        sent_starts = list(map(lambda x: int(''.join(x)), args['sent_starts']))
        # sent_starts = list(map(lambda x: int(x), [item for sublist in args['sent_starts'] for item in sublist]))
        ner_tags = list(map(lambda x: ''.join(x), args['ner_tags']))
        # ner_tags = [''.join(item) for sublist in args['ner_tags'] for item in sublist]
        print(token_list)
        print(sent_starts)
        print(ner_tags)
        rels = relex.predict(token_list, sent_starts, ner_tags)
        print(rels)
        try:
            return {
                "relations": rels,
            }
        except Exception as e:
            relex_space.abort(400, e.__doc__, status="Could not perform prediction", statusCode="400")

# ["Wegen","Hypocalciaemie","stationaer","auf","der","134","vom","22.7",".","bis","3.8.04","."]
# ["O","B-Medical_condition","B-Treatment","O","O","O","O","B-Time_information","I-Time_information","I-Time_information","I-Time_information","O"]
# token_list=Wegen&token_list=Hypocalciaemie&token_list=stationaer&token_list=auf&token_list=der&token_list=134&token_list=vom&token_list=22.7&token_list=.&token_list=bis&token_list=3.8.04&token_list=.&token_list=Gutes&token_list=Befinden&token_list=,&token_list=keine&token_list=Kraempfe&token_list=,&token_list=keine&token_list=GI-Symptome&token_list=,&token_list=kein&token_list=epigastrischen&token_list=Schmerzen&token_list=,&token_list=Stuhlgang&token_list=normal&token_list=.&sent_starts=11&ner_tags=O&ner_tags=B-Medical_condition&ner_tags=B-Treatment&ner_tags=O&ner_tags=O&ner_tags=O&ner_tags=B-Time_information&ner_tags=I-Time_information&ner_tags=I-Time_information&ner_tags=I-Time_information&ner_tags=I-Time_information&ner_tags=O&ner_tags=B-State_of_health&ner_tags=I-State_of_health&ner_tags=O&ner_tags=O&ner_tags=B-Medical_condition&ner_tags=O&ner_tags=O&ner_tags=B-Medical_condition&ner_tags=O&ner_tags=O&ner_tags=B-Local_specification&ner_tags=B-Medical_condition&ner_tags=O&ner_tags=B-Process&ner_tags=B-State_of_health&ner_tags=O
