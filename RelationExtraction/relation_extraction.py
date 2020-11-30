from itertools import combinations

import spacy
from spacy.tokens import Span, Doc
from flair.models import TextClassifier
from flair.data import Sentence, Token
import pickle


class RelationExtraction:
    name = 'relation_extraction'

    def __init__(self, nlp, model_file):
        self.nlp = nlp
        self.clf = TextClassifier.load_from_file(model_file)
        # self.concept_map = pickle.load(open(dict_map_path, "rb"))
        self.concept_map = {'_UNK': [0], 'Dosing': [1], 'State_of_health': [2], 'Measurement': [3], 'Negation': [4],
                            'Treatment': [5], 'Medical_condition': [6], 'Process': [7], 'Medication': [8],
                            'Person': [9], 'Medical_device': [10], 'Time_information': [11], 'Body_part': [12],
                            'DiagLab_Procedure': [13], 'Local_specification': [14], 'Biological_chemistry': [15],
                            'Structure_element': [16], 'Biological_parameter': [17], 'Body_Fluid': [18], 'Type': [19],
                            'Speculation': [20], 'Tissue': [21], 'Degree': [22], 'Medical_specification': [23],
                            'Temporal_course': [24]}
        for label in self.clf.label_dictionary.get_items():
            self.nlp.vocab.strings.add(label)
            # split = tag.split('-')
            # add tags without iob prefix to string store
            # if len(split) == 2:
            #    self.nlp.vocab.strings.add(split[1])

        Doc.set_extension('rels', default=[], force=True)

    @staticmethod
    def _group_entities_by_sentence(doc):
        sentence_ends = [ent.end for ent in doc.sents]

        entity_groups = []
        entity_group = []
        current_sentence = 0
        for entity in doc.ents:
            if entity.end > sentence_ends[current_sentence]:
                if entity_group:
                    entity_groups.append(entity_group)
                    entity_group = []
                current_sentence += 1

            entity_group.append(entity)

        if entity_group:
            entity_groups.append(entity_group)

        return entity_groups
    
    @staticmethod
    def _prepare_sentence(sent, entity1, entity2, map_dict):
        def add_offset_to_sentence(sentence, span, tag):
            start, end = span
            for i, token in enumerate(sentence.tokens):
                if i >= end:
                    token.add_tag(tag, (i + 1) - end)
                elif i < start:
                    token.add_tag(tag, i - start)
                else:
                    token.add_tag(tag, 0)

        def add_concept_to_sentence(sentence, span, tag, tag_value):
            start, end = span
            for i, token in enumerate(sentence.tokens):
                if i >= end:
                    token.add_tag(tag, 0)
                elif i < start:
                    token.add_tag(tag, 0)
                else:
                    token.add_tag(tag, tag_value)

        sentence: Sentence = Sentence()
        for token in sent:
            sentence.add_token(Token(token.text))

        #print(entity1.label_, entity2.label_)
        sent_offset = sent.start
        add_offset_to_sentence(sentence, (entity1.start - sent_offset, entity1.end - sent_offset), tag='offset_e1')
        add_offset_to_sentence(sentence, (entity2.start - sent_offset, entity2.end - sent_offset), tag='offset_e2')

        add_concept_to_sentence(sentence, (entity1.start - sent_offset, entity1.end - sent_offset), tag='concept_1', tag_value=map_dict[entity1.label_][0])
        add_concept_to_sentence(sentence, (entity2.start - sent_offset, entity2.end - sent_offset), tag='concept_2', tag_value=map_dict[entity2.label_][0])

        return sentence
    
    def __call__(self, doc):
        def swap_entities(relation):
            return relation[-6:-1].lower() == 'e2,e1'
        
        def relation_name(relation):
            return relation[:-7]
        
        def negative_relation(relation):
            return relation.lower().startswith('not_')
            
        entity_groups = self._group_entities_by_sentence(doc)
        
        sentences = []
        entity_combinations = []
        for sentence, entities in zip(doc.sents, entity_groups):
            for entity_left, entity_right in combinations(entities, r=2):
                sentences.append(self._prepare_sentence(sentence, entity_left, entity_right, self.concept_map))
                entity_combinations.append((entity_left, entity_right))

        #print(self.concept_map)
        #print(self.concept_map)
        pred_sentences = self.clf.predict(sentences)
        
        relations = []
        for sent, (ent1, ent2) in zip(pred_sentences, entity_combinations):
            relation = sent.labels[0].name
            
            if not negative_relation(relation):
                if swap_entities(relation):
                    relations.append((ent2, ent1, relation_name(relation)))
                else:
                    relations.append((ent1, ent2, relation_name(relation)))
        
        doc._.rels = relations

        return doc
