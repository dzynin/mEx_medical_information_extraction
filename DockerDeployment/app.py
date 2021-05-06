import werkzeug

werkzeug.cached_property = werkzeug.utils.cached_property

from flask import Flask, request
from flask_restplus import Api, Resource, fields
from flair.data import Sentence
from flair.models import SequenceTagger
from RelationExtraction import RelationExtractionModel

from flairrelex.models import TextClassifier
import torch

# RelationExtraction: RelationExtractionModel = RelationExtractionModel()
# NERTagger: SequenceTagger = SequenceTagger.load(model="ner.pt")
# POSTagger: SequenceTagger = SequenceTagger.load(model="pos.pt")


flask_app = Flask(__name__)
app = Api(app=flask_app,
          version="1.0",
          title="mEx Medical SequenceTagger API (NER & POS)",
          description="This API offers an interface to use NER & POS models for text prediction",
          contact="Ammer Ayach",
          contact_email="amay01@dfki.de")

if __name__ == '__main__':
    clf = TextClassifier.load_from_file('flairrelex.pt')

    allowed_pairs = ['State_of_health-Process', 'Measurement-Process', 'Measurement-Medical_condition',
                     'Dosing-Treatment', 'Time_information-Medical_condition', 'Time_information-Treatment',
                     'Treatment-Medication', 'Medical_condition-Local_specification', 'DiagLab_Procedure-Body_part',
                     'DiagLab_Procedure-Medical_condition', 'DiagLab_Procedure-Measurement',
                     'Body_part-Medical_condition', 'Medical_device-Treatment', 'Medication-Dosing',
                     'Biological_chemistry-Measurement', 'Medical_specification-Medical_condition',
                     'Treatment-Medical_specification', 'DiagLab_Procedure-Biological_chemistry']

    concept_map = {'_UNK': 0, 'State_of_health': 1, 'Measurement': 2, 'Medical_condition': 3, 'Process': 4,
                   'Medication': 5, 'Dosing': 6, 'Treatment': 7, 'Person': 8, 'Time_information': 9,
                   'DiagLab_Procedure': 10, 'Local_specification': 11, 'Biological_chemistry': 12,
                   'Body_part': 13, 'Medical_device': 14, 'Biological_parameter': 15, 'Body_Fluid': 16,
                   'Medical_specification': 17}

    tagger_tag_dictionary = ['<unk>', 'O', 'B-Time_information', 'I-Time_information', 'B-Treatment',
                             'B-Body_part', 'B-State_of_health', 'B-Medical_condition', 'I-State_of_health',
                             'B-DiagLab_Procedure', 'I-DiagLab_Procedure', 'B-Medication', 'I-Treatment',
                             'B-Biological_chemistry', 'B-Measurement', 'I-Measurement',
                             'B-Biological_parameter', 'B-Medical_specification', 'B-Person', 'I-Body_part',
                             'I-Medical_condition', 'B-Process', 'B-Local_specification', 'B-Dosing',
                             'I-Dosing', 'I-Biological_chemistry', 'I-Medication', 'I-Person',
                             'I-Biological_parameter', 'I-Process', 'I-Local_specification', 'B-Body_Fluid',
                             'B-Medical_device', 'I-Medical_specification', 'I-Medical_device', 'I-Body_Fluid',
                             '<START>', '<STOP>']

    ababa = clf.state_dict()

    clf.state_dict()['allowed_pairs'] = allowed_pairs
    cacaca = clf.state_dict()

    print()


    #clf.label_dictionary.add_item('allowed_pairs')

    #clf.label_dictionary.add_item(concept_map)
    #clf.label_dictionary.add_item(tagger_tag_dictionary)


    print()

    clf.save('flairrelex2.pt')