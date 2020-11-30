import spacy
from spacy.tokens import Span, Doc
from flair.models import SequenceTagger
from flair.data import Sentence, Token
import json
import requests


class NER:
    name = 'ner'


    def __init__(self, nlp):
        self.nlp = nlp
        self.req_address = "http://127.0.0.1:5001/predict"
        # self.tagger = SequenceTagger.load_from_file(model_file)
        self.tagger_tag_dictionary = ['<unk>', 'O', 'B-Time_information', 'I-Time_information', 'B-Treatment',
                                      'B-Body_part', 'B-State_of_health', 'B-Medical_condition', 'I-State_of_health',
                                      'B-DiagLab_Procedure', 'I-DiagLab_Procedure', 'B-Medication', 'I-Treatment',
                                      'B-Biological_chemistry', 'B-Measurement', 'I-Measurement',
                                      'B-Biological_parameter', 'B-Medical_specification', 'B-Person', 'I-Body_part',
                                      'I-Medical_condition', 'B-Process', 'B-Local_specification', 'B-Dosing',
                                      'I-Dosing', 'I-Biological_chemistry', 'I-Medication', 'I-Person',
                                      'I-Biological_parameter', 'I-Process', 'I-Local_specification', 'B-Body_Fluid',
                                      'B-Medical_device', 'I-Medical_specification', 'I-Medical_device', 'I-Body_Fluid',
                                      '<START>', '<STOP>']


        # for tag in self.tagger.tag_dictionary.get_items():

        for tag in self.tagger_tag_dictionary:
            self.nlp.vocab.strings.add(tag)
            split = tag.split('-')
            # add tags without iob prefix to string store
            if len(split) == 2:
                self.nlp.vocab.strings.add(split[1])

    def merge_iob_spans(self, doc, spans, tags):
        def create_entity_span(tokens):
            if len(tokens) > 1:
                start = tokens[0].start
                end = tokens[-1].end
                return Span(doc, start, end, label=tokens[0].label)
            elif len(tokens) == 1:
                return tokens[0]
            return None
        
        entity_tokens = []
        spans_for_merge = []
        for i, (span, tag) in enumerate(zip(spans, tags)):
            if tag == 'O':
                continue
            split = tag.split('-')
            if len(split) != 2 or split[0] not in ['I', 'B']:
                return False
            if split[0] == 'B':
                merge_span = create_entity_span(entity_tokens)
                if merge_span:
                    spans_for_merge.append(merge_span)
                entity_tokens = [span]
            elif tags[i - 1][1:] == tag[1:]:
                entity_tokens.append(span)
        # merge remaining entity at the end
        merge_span = create_entity_span(entity_tokens)
        if merge_span:
            spans_for_merge.append(merge_span)
        return spans_for_merge
        #for span in spans_for_merge:
        #    span.merge()
    
    def __call__(self, doc):

        # TODO: use a sentencizer or not?
        # TODO: process all sentences in one batch on GPU
        for doc_sentence in doc.sents:
            #filtered_doc_sentence = [token for token in doc_sentence if not token.is_punct and not token.is_space]
            filtered_doc_sentence = doc_sentence

            json_data = []

            # if still token remaining in sentence
            if filtered_doc_sentence:
                sentence = Sentence()
                for token in filtered_doc_sentence:
                    sentence.add_token(Token(token.text))
                    json_data.append(token.text)

                json_obj = json.dumps(json_data)

                r = requests.post(self.req_address, json=json_obj)

                tags_res = r.json()

                spans = []
                tags = []
                for doc_token, tag in zip(filtered_doc_sentence, tags_res):
                    start = doc_token.i
                    end = start + 1

                    #tag = tagged_token.get_tag('ner')

                    if tag != 'O':
                        _, label = tag.split('-')
                        span = Span(doc, start, end, label=self.nlp.vocab.strings[label])
                        spans.append(span)
                        tags.append(tag)
                        # doc.ents = list(doc.ents) + [span]
                doc.ents = list(doc.ents) + self.merge_iob_spans(doc, spans, tags)

                """
                tagged_sentences = self.tagger.predict(sentence)

                spans = []
                tags = []
                for doc_token, tagged_token in zip(filtered_doc_sentence, tagged_sentences[0]):
                    start = doc_token.i
                    end = start + 1

                    tag = tagged_token.get_tag('ner')

                    if tag != 'O':
                        _, label = tag.split('-')
                        span = Span(doc, start, end, label=self.nlp.vocab.strings[label])
                        spans.append(span)
                        tags.append(tag)
                        #doc.ents = list(doc.ents) + [span]
                        
                doc.ents = list(doc.ents) + self.merge_iob_spans(doc, spans, tags)
                """

        return doc
