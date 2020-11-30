import spacy

from NamedEntityRecognition.named_entity_recognition import NER
from RelationExtraction.relation_extraction import RelationExtraction


class MedicalIEPipeline:
    _MEDICAL_IE_PIPELINE = None

    @staticmethod
    def get_pipeline():
        if MedicalIEPipeline._MEDICAL_IE_PIPELINE is None:
            pipeline = spacy.load('de_core_news_sm', disable=['tagger', 'ner', 'textcat'])

            named_entity_recognition = NER(pipeline)
            relation_extraction = RelationExtraction(pipeline, model_file='../Resources/relation_extraction_mex_model(default_word_relative_concept_embeddings).pt')

            pipeline.add_pipe(named_entity_recognition)
            pipeline.add_pipe(relation_extraction)

            MedicalIEPipeline._MEDICAL_IE_PIPELINE = pipeline

        return MedicalIEPipeline._MEDICAL_IE_PIPELINE

    @staticmethod
    def get_annotated_document(text):
        return MedicalIEPipeline.get_pipeline()(text)
