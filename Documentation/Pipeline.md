# Pipeline
## Pipeline Architecture:
Replacing Spacy's NLP Pipeline components with our models enable us to apply the models sequentially feeding the output of the NER model into RelEx.
### Code:
 ```
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

 ```
## Apply Models & Generate Brat Data:
The **generate_brat_data.py** script takes in text files, applies models, convert predictions into Brat Format.
### Usage:
 ```
# Make sure NER server (app.py) is on
conda activate RelEx
python generate_brat_data.py
 ```