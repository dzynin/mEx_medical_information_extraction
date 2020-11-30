import spacy
import configparser
import os
import re
import string
from bs4 import BeautifulSoup

config = configparser.ConfigParser()
config.read('tok_sent_recal_off.ini')


class TextUtilities(object):
    def __init__(self):
        self.old_data_path = config['ARGUMENTS']['old_data_input_path']
        self.output_path = config['ARGUMENTS']['output_path']

        if self.old_data_path[-1] != "/":
            self.old_data_path += "/"

        if self.output_path[-1] != "/":
            self.output_path += "/"

        self.xmi_or_spacy = int(config['ARGUMENTS']['xmi_or_spacy'])

        if self.xmi_or_spacy == 1:
            self.nlp = spacy.load("de_core_news_lg")

    @staticmethod
    def get_valid_file_names(input_folder):
        #os.chdir(input_folder)
        file_names_no_extension = list(
            set(list(map(lambda x: x.replace('.ann', '').replace('.txt', ''), os.listdir(input_folder)))))

        valid_file_names = []
        for file in file_names_no_extension:
            if os.path.isfile(input_folder + file + '.txt') and os.path.isfile(input_folder + file + '.ann'):
                valid_file_names.append(file)

        return valid_file_names

    @staticmethod
    def text_sentence_offsets(text):
        sent_offs = []
        sentences_array = text.split("\n")[0:-1] if not text.split("\n")[-1] else text.split("\n")
        sentences_array = list(map(lambda x: x.strip(), sentences_array))

        index = -1
        for i in range(len(sentences_array)):
            if i == 0:
                sent_offs.append((0, len(sentences_array[i])))
                index = len(sentences_array[i])
            else:
                sent_offs.append((index + 1, index + 1 + len(sentences_array[i])))
                index = index + 1 + len(sentences_array[i])

        return sent_offs

    def sentence_split_tokenize_text(self, filename, text_data):
        doc = self.nlp(text_data.replace("\n", ""))
        new_txt = open(self.output_path + filename + '.txt', 'w+')
        sents = [sent for sent in doc.sents]

        for i in range(len(sents)):
            new_txt.write(" ".join([token.text for token in sents[i]]))
            if i != len(sents) - 1:
                new_txt.write('\n')

        new_txt.close()

    def sentence_split_tokenize_text_jcore_conll(self, filename, old_text):
        conll_file = open(self.output_path + 'conll/' + filename.replace('_brat', '') + '.CONLL', encoding='utf8').read().split('\n\n')
        conll_file = list(map(lambda x: x.split('\n'), list(filter(lambda x: x != '', conll_file))))
        conll_file = '\n'.join(list(map(lambda x: ' '.join(x), [list(map(lambda x: x.split('\t')[1], i)) for i in conll_file])))
        new_text = open(self.output_path + filename + '.txt', 'w+')
        new_text.write(conll_file)
        new_text.close()

    def sentence_split_tokenize_text_jcore_xmi(self, filename, old_text):
        os.chdir(self.output_path + 'xmi/')
        soup = BeautifulSoup(open((re.sub("_brat", "", filename)) + ".xmi"), 'lxml-xml')

        sentence_end = []
        for doc in soup.find_all('Sentence'):
            if doc.has_attr('end'):
                sentence_end.append(doc.attrs['end'])
        text = ""
        # tokens = []
        for document in soup.find_all('STTSMedPOSTag'):
            begin = ""
            end = ""
            if document.has_attr('begin'):
                begin = document.attrs['begin']

            if document.has_attr('end'):
                end = document.attrs['end']

            # print("-->"+old_text[int(begin):int(end)]+"<--")

            if end in sentence_end:
                # tokens.append(old_text[int(begin):int(end)]+'\n')
                text += old_text[int(begin):int(end)] + '\n'
            else:
                # tokens.append(old_text[int(begin):int(end)])
                text += old_text[int(begin):int(end)] + ' '

        # print(text)
        # tokens = " ".join(tokens)
        # print(tokens)
        new_text = open(self.output_path + filename + '.txt', 'w+')
        new_text.write(text)
        new_text.close()

    def extract_ann_data(self, ann_file):
        annotations = []
        relations = []
        offsets_all = []
        for x_line in ann_file:
            x_line = re.sub("\n", "", x_line)
            if re.match("^T[0-9]", x_line):
                x_line = x_line.split("\t")
                id = x_line[0]
                token = x_line[-1]
                label = x_line[1].split(" ")[0]
                offsets = [(int(i[0]), int(i[1])) for i in
                           list(map(lambda x: x.split(","), ",".join(x_line[1].split(" ")[1:]).split(";")))]
                offsets_all.append(offsets)
                annotations.append(self.AnnotationEntity(id, label, offsets, token))
            elif re.match("^R", x_line):
                x_line = x_line.split('\t')
                id = x_line[0]
                rel_name = x_line[1].split(' ')[0]
                arg1 = x_line[1].split(' ')[1].split(':')[1]
                arg2 = x_line[1].split(' ')[2].split(':')[1]
                relations.append(self.RelationEntity(id, rel_name, arg1, arg2))

        for ann in annotations:
            for rel in relations:
                if ann.id == rel.arg1 or ann.id == rel.arg2:
                    ann.add_relation(rel)

        return annotations, offsets_all

    @staticmethod
    def in_offset(offset, offs):
        for i in offs:
            if offset == i[0]:
                return True, 0
            if offset == i[1] - 1:
                return True, 1
        return False, -1

    def recalc_annotation(self, org_text, new_text):
        char_map = {}
        index = 0
        for i in range(len(org_text)):
            if org_text[i] not in string.whitespace:
                for j in range(len(new_text[index:])):
                    if new_text[index:][j] == org_text[i]:
                        char_map[i] = index + j
                        index = index + j + 1
                        break

                        """
                        if self.in_offset(i, ann_offsets)[1] == 0:
                            char_map[i] = index + j
                            index = index + j + 1
                            break
                        if self.in_offset(i, ann_offsets)[1] == 1:
                            char_map[i + 1] = index + j + 1
                            index = index + j + 1
                            break
                        """

        # FOR TESTING
        """
        for i in char_map.keys():
            try:
                assert "".join(org_text[0:i].split()) == "".join(new_text[char_map[0]:char_map[i]].split())
            except:
                print("old:  ", "".join(org_text[0:i].split()))
                print("new:  ", "".join(new_text[char_map[0]:char_map[i]].split()))
                print("-------------\n")
        """
        return char_map

    @staticmethod
    def tupel_in_sent(tup, sents):
        for sent in sents:
            if sent[0] <= tup[0] and tup[1] <= sent[1]:
                return True, sent
        return False, None

    def valid_set_tups(self, tups, sents):
        id_dict = {}
        for tup in tups:
            if self.tupel_in_sent(tup, sents)[1] not in id_dict.keys():
                id_dict[self.tupel_in_sent(tup, sents)[1]] = [tup]
            else:
                id_dict[self.tupel_in_sent(tup, sents)[1]].append(tup)

        if len(set(list(id_dict.keys()))) == 1:
            return True, id_dict
        else:
            return False, id_dict

    @staticmethod
    def break_tuple(tup, sents):
        new_offs = []
        for sent in sents:
            if sent[0] <= tup[0] <= sent[1]:
                new_offs.append((tup[0], sent[1]))
            if sent[0] <= tup[1] <= sent[1]:
                new_offs.append((sent[0], tup[1]))
        return new_offs



    def update_annotation_offsets(self, anns, char_map, new_sent_offs, new_text):
        new_anns = []
        t_id = 0
        for ann in anns:
            new_offs = [(char_map[i[0]], char_map[i[1] - 1] + 1) for i in ann.offsets]
            for tup in new_offs:
                if not self.tupel_in_sent(tup, new_sent_offs)[0]:
                    break_offs = self.break_tuple(tup, new_sent_offs)
                    new_offs.remove(tup)
                    new_offs.extend(break_offs)
            res = self.valid_set_tups(new_offs, new_sent_offs)
            if res[0]:
                new_anns.append(self.AnnotationEntity("T" + str(t_id), ann.label, sorted(new_offs, key=lambda x: x[0]),
                                                      " ".join([new_text[i[0]:i[1]] for i in
                                                                sorted(new_offs, key=lambda x: x[0])])))
                new_anns[-1].set_old_id(ann.id)
                new_anns[-1].relations = ann.relations
                t_id += 1
            else:
                for i in res[1].keys():
                    new_anns.append(
                        self.AnnotationEntity("T" + str(t_id), ann.label, sorted(res[1][i], key=lambda x: x[0]),
                                              " ".join([new_text[j[0]:j[1]] for j in
                                                        sorted(res[1][i], key=lambda x: x[0])])))
                    new_anns[-1].set_old_id(ann.id)
                    new_anns[-1].relations = ann.relations
                    t_id += 1

        return new_anns

    @staticmethod
    def check_if_relation_in_same_sentence(ann1, ann2, new_sent_offs):
        same = []
        for off in new_sent_offs:
            for off1 in ann1.offsets:
                if off[0] <= off1[0] and off1[1] <= off[1]:
                    same.append(off)
            for off2 in ann2.offsets:
                if off[0] <= off2[0] and off2[1] <= off[1]:
                    same.append(off)
        if len(set(same)) == 1:
            return True
        else:
            return False


    def update_relations(self, new_anns, new_sent_offs):
        new_rels = []
        for ann in new_anns:
            if ann.relations:
                for rel in ann.relations:
                    for ann2 in new_anns:
                        if ann.id != ann2.id and (ann2.old_id == rel.arg1 or ann2.old_id == rel.arg2):
                            if self.check_if_relation_in_same_sentence(ann, ann2, new_sent_offs):
                                if rel.arg1 == ann.old_id:
                                    rel.arg1 = ann.id

                                if rel.arg2 == ann.old_id:
                                    rel.arg2 = ann.id

                                if rel.arg1 == ann2.old_id:
                                    rel.arg1 = ann2.id

                                if rel.arg2 == ann2.old_id:
                                    rel.arg2 = ann2.id

                                new_rels.append(rel)

        new_rels = list(dict.fromkeys(new_rels))

        return new_rels

    def create_new_brat_file(self, anns, rels, file):
        new_ann = open(self.output_path + file + '.ann', 'w+')
        for i in range(len(anns)):
            new_ann.write(anns[i].id + "\t" + anns[i].label + " " + ";".join(
                [str(j[0]) + ' ' + str(j[1]) for j in anns[i].offsets]) + "\t" + anns[i].token + "\n")

        for i in range(len(rels)):
            new_ann.write(rels[i].id + "\t" + rels[i].relation_name + " arg1:"+rels[i].arg1+" arg2:"+rels[i].arg2)
            if i != len(anns) - 1:
                new_ann.write('\n')

        new_ann.close()

    class AnnotationEntity(object):
        def __init__(self, id, label, offsets, token):
            self.id = id
            self.label = label
            self.offsets = offsets
            self.token = token
            self.relations = []
            self.old_id = -1

        def add_relation(self, relation):
            self.relations.append(relation)

        def set_old_id(self, id):
            self.old_id = id

    class RelationEntity(object):
        def __init__(self, id, rel_name, arg1, arg2):
            self.id = id
            self.relation_name = rel_name
            self.arg1 = arg1
            self.arg2 = arg2
            self.active = True

            
def main():
    txtutl = TextUtilities()
    filenames = txtutl.get_valid_file_names(txtutl.old_data_path)
    #os.chdir(txtutl.old_data_path)
    for file in filenames:
        print(file)
        old_text = open(txtutl.old_data_path + file + '.txt', encoding='utf8').read()

        if txtutl.xmi_or_spacy == 1:
            txtutl.sentence_split_tokenize_text(file, old_text)
        elif txtutl.xmi_or_spacy == 0:
            txtutl.sentence_split_tokenize_text_jcore_conll(file, old_text)
        else:
            pass

        new_text = open(txtutl.output_path + file + '.txt', encoding='utf8').read()
        new_sent_offs = txtutl.text_sentence_offsets(new_text)
        anns, offs = txtutl.extract_ann_data(open(txtutl.old_data_path + file + '.ann', encoding='utf8'))

        char_map = txtutl.recalc_annotation(old_text, new_text)
        new_anns = txtutl.update_annotation_offsets(anns, char_map, new_sent_offs, new_text)
        new_rels = txtutl.update_relations(new_anns, new_sent_offs)
        txtutl.create_new_brat_file(new_anns, new_rels, file)


if __name__ == '__main__':
    main()
