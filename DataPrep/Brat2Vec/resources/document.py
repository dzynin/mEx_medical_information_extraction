from typing import List


class Span:
    def __init__(self, offsets):
        self.start = offsets[0]
        self.end = offsets[1]

    def is_overlapping(self, span):
        return (self.start < span.end) and (self.end > span.start)

    def is_within(self, span):
        return span.start <= self.start < self.end <= span.end


class Token(Span):
    def __init__(self, offsets, document_text):
        super().__init__(offsets)
        self.text = document_text[self.start:self.end]
        self.embedding_index = None

    def set_embedding_index(self, embedding_index):
        self.embedding_index = embedding_index


def __compute_distance_to_range__(index, start_index, end_index):
    if index < start_index:
        return index - start_index
    elif index > end_index:
        return index - end_index
    else:
        return 0


class Concept(Span):
    def __init__(self, concept_type: str, offsets, text: str):
        super().__init__(offsets)
        self.concept_type = concept_type
        self.text = text
        self.tokens = []

    def add_token(self, token):
        self.tokens.append(token)

    def position_distances_to(self, tokens):
        start_index = tokens.index(self.tokens[0])
        end_index = tokens.index(self.tokens[-1])

        return [__compute_distance_to_range__(index, start_index, end_index) for index, _ in enumerate(tokens)]


class Relation(Span):
    def __init__(self, relation_type: str, first_argument: Concept, second_argument: Concept):
        self.first_argument = first_argument
        self.second_argument = second_argument
        self.argument_offsets = [min(self.first_argument.start, self.second_argument.start),
                                 max(self.first_argument.end, self.second_argument.end)]
        super().__init__(self.argument_offsets)
        self.relation_type = relation_type

    def equals(self, relation):
        return (self.first_argument == relation.first_argument and
                self.second_argument == relation.second_argument)


class Sentence(Span):
    def __init__(self, offsets, document_text):
        super().__init__(offsets)
        self.concepts = []
        self.relations = []
        self.tokens = []
        self.text = document_text[self.start:self.end]

    def add_concept(self, concept: Concept):
        self.concepts.append(concept)

    def add_relation(self, relation: Relation):
        self.relations.append(relation)

    def add_tokens_and_fix_offsets(self, tokens: List[Token]):
        for token in tokens:
            token.start += self.start
            token.end += self.start
            self.tokens.append(token)

    def has_relation_arguments(self, relation: Relation):
        return ((relation.first_argument in self.concepts) and
                (relation.second_argument in self.concepts))


class Document:
    def __init__(self, file_name: str, text: str, concepts: List[Concept], relations: List[Relation]):
        self.file_name = file_name
        self.text = text
        self.concepts = concepts
        self.relations = relations
        self.sentences = []

    def add_sentence(self, sentence: Sentence):
        self.sentences.append(sentence)
