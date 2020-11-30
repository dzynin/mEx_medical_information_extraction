# coding=utf-8

import os
import re
import os.path
import nltk
import copy
import configparser
import argparse
import pickle

from DataPrep.Brat2Vec.resources.document import Token
from DataPrep.Brat2Vec.resources.vocabulary import Vocabulary


# from vocabularyGB import VocabularyGB

# this script converts brat input files into the format appropriate for Marc's CNN model.
# We assume that the brat data is already enriched with positive and negative examples
# considers only intra-sentence relations and ignores the rest
# we expect the input text to be tokenized!!!
# better you provide already single sentences as input the nltk sentence splitter is NOT awesome!

# conda info --envs
# to see which ENV-> choose:
# source activate RE_CNN

# python brat2vec.py --c configs/brat2vec_spanish.cfg

# >> you need to install this to use metamap on python level: https://github.com/AnthonyMRios/pymetamap (including the original metamap version)

class Brat2Vec:

    def __init__(self, config):

        self.sentence_split = int(config['Preprocessor'].get('sentence_split'))
        self.out_file_train = config['Files'].get('out_file_train')
        self.out_file_test = config['Files'].get('out_file_test')

        self.tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')
        self.relation_sent_set = []

        self.preprocessor_dir = config['Files'].get('preprocessor_dir')
        os.makedirs(self.preprocessor_dir, exist_ok=True)
        self.vocabulary_dir = os.path.join(self.preprocessor_dir, "vocabulary")
        self.vocabulary_concept_id = os.path.join(self.preprocessor_dir, "vocabulary_concept_id")
        self.umls_vocabulary_dir = os.path.join(self.preprocessor_dir, "umls_vocabulary")
        self.max_sentence_length = config['Preprocessor'].getint('max_sentence_length', 10000)

        # Vocabulary parameters
        self.unknown_token = config['Preprocessor'].get('unknown_token')
        self.padding_token = config['Preprocessor'].get('padding_token')

        self.use_graphembed = config['Preprocessor'].get('use_graphembed', 0)
        self.embeddings_file = config['Files'].get('embeddings_file')

        self.min_word_frequency = config['Preprocessor'].getint('min_word_frequency', 1)

        self.embeddings_dim = config['Model'].getint('embeddings_dim', 300)
        self.graphembedding_dim = config['Model'].getint('graphembedding_dim', 300)
        self.embeddings_norm = config['Model'].getfloat('embeddings_norm', 0.25)

        self.ner_labels = int(config['Preprocessor'].getint('include_ner_labels', 0))
        self.ffirst_occ = int(config['Preprocessor'].getint('always_first_first_occurrence', 0))
        self.max_word_dist = int(config['Model'].getint('max_word_dist', 50))

        self.concept_labels = {}

        self.all_concepts_set = {}
        self.all_concepts_set["_UNK"] = [0]

        self.concepts_ids = 1
        self.add_concept_embed = int(config['Preprocessor'].get('add_concept_embeddings'))


    def read_brat(self, brat_folder):
        print("read brat:", brat_folder)

        relation_sent_set = []
        # we expect that .ann file exist for each .txt file in the same folder (even if empfty)
        # Iterate through the filenames in the given Directory
        for filename in os.listdir(brat_folder):
            # If txt file
            if filename.endswith("txt"):
                print("")
                print("")
                print("''''''''''''''''''''")
                print("file:", filename)

                file_root = re.sub("\.txt", "", filename)
                concept = {}
                relations = []
                txt_set = []
                # read in annotations
                file = open(brat_folder + "/" + file_root + ".ann")
                # iterate through the current brat file
                for x_line in file:

                    if re.match("^T[0-9]", x_line):
                        x_line = re.sub("\n", "", x_line)
                        f_split = x_line.split("\t")

                        if len(f_split) > 2:

                            s_split = f_split[1].split(" ")

                            if len(s_split) > 3:
                                s_split[2] = s_split[len(s_split) - 1]

                            if s_split[0] != "Kommentar":

                                # Gather all the concepts in both train and test data A code
                                if s_split[0] not in self.all_concepts_set:
                                    self.all_concepts_set[s_split[0]] = [self.concepts_ids]
                                    self.concepts_ids += 1

                                concept[f_split[0]] = [s_split[0], int(s_split[1]), int(s_split[2]), f_split[2],
                                                       s_split[2]]

                    elif re.match("^R[0-9]", x_line):
                        x_line = re.sub("\n", "", x_line)
                        f_split = x_line.split("\t")

                        if len(f_split) > 1:
                            s_split = f_split[1].split(" ")
                            if len(s_split) > 3:
                                s_split[2] = s_split[len(s_split) - 1]
                            relations.append(
                                [s_split[0], (s_split[1].split(":"))[1], (s_split[2].split(":"))[1], 0, x_line[0]])


                    elif re.match("^E[0-9]", x_line):
                        x_line = re.sub("\n", "", x_line)
                        f_split = x_line.split("\t")

                        if len(f_split) > 1:
                            s_split = f_split[1].split(" ")
                            if len(s_split) > 3:
                                s_split[2] = s_split[len(s_split) - 1]
                            relations.append(
                                [(s_split[0].split(":"))[0], (s_split[0].split(":"))[1], (s_split[1].split(":"))[1], 0,
                                 x_line[0]])

                file = open(brat_folder + "/" + file_root + ".txt")
                line_offset = 0
                for x_line in file:
                    txt_set.append([x_line, int(line_offset), int(line_offset + len(x_line))])
                    line_offset += len(x_line)

                # apply sentence splitting
                sent_set = []
                if self.sentence_split == 1:
                    #print(">sentence_split")
                    for l in txt_set:
                        sent_offset = self.tokenizer.span_tokenize(l[0])
                        offset_start = l[1]

                        for new_sent_o in sent_offset:
                            txt_substr = l[0][new_sent_o[0]:new_sent_o[1]]
                            sent_set.append([txt_substr, offset_start + new_sent_o[0], offset_start + new_sent_o[1]])

                else:
                    sent_set = txt_set

                for l in sent_set:

                    sent = re.sub("\n", "", l[0])
                    l_con = []
                    # now find for each line all concepts:
                    for e in concept.keys():
                        con = concept[e]
                        if int(l[1]) <= int(con[1]) and int(l[2]) >= int(con[2]):
                            l_con.append([e, con[1], con[2], con[0], con[3]])

                    for id1 in range(len(l_con)):
                        for id2 in range(len(l_con)):
                            if id1 != id2:

                                for r in relations:
                                    if (l_con[id1][0] == r[1] and l_con[id2][0] == r[2]):
                                        print_rel = 0
                                        r[3] = 1  # mark relation as used
                                        diff = l[1]

                                        inverse_rel = 0
                                        rel_extend = ""
                                        if self.ffirst_occ == 1:
                                            # alsways put the first occurrence of entity to position 1
                                            if l_con[id1][1] > l_con[id2][1]:
                                                con1 = copy.deepcopy(l_con[id2])
                                                con2 = copy.deepcopy(l_con[id1])
                                                inverse_rel = 1
                                            else:
                                                con1 = copy.deepcopy(l_con[id1])
                                                con2 = copy.deepcopy(l_con[id2])
                                        else:
                                            con1 = copy.deepcopy(l_con[id1])
                                            con2 = copy.deepcopy(l_con[id2])

                                        if inverse_rel == 1:
                                            rel_extend = "(e2,e1)"
                                        else:
                                            rel_extend = "(e1,e2)"

                                        con1[1] = con1[1] - diff
                                        con1[2] = con1[2] - diff
                                        con2[1] = con2[1] - diff
                                        con2[2] = con2[2] - diff

                                        concept_label1 = con1[3]
                                        if not concept_label1 in self.concept_labels:
                                            self.concept_labels[concept_label1] = len(self.concept_labels.keys()) + 1
                                        concept_label2 = con2[3]
                                        if not concept_label2 in self.concept_labels:
                                            self.concept_labels[concept_label2] = len(self.concept_labels.keys()) + 1

                                        # identify word pos for given offsets
                                        word_tokenizer = nltk.tokenize.WordPunctTokenizer()
                                        # token_offsets = word_tokenizer.span_tokenize(l[0])
                                        # print (">", token_offsets)
                                        # for x in token_offsets:
                                        #    print ("<<<", x)

                                        e1 = self.get_word_pos(word_tokenizer.span_tokenize(l[0]), con1[1], con1[2])
                                        e2 = self.get_word_pos(word_tokenizer.span_tokenize(l[0]), con2[1], con2[2])

                                        # print (filename, l[0], l[1] - diff, l[2] - diff, con1, e1, con2, e2, r)
                                        e1.append(con1[3])
                                        e1.append(con1[4])
                                        e1.append(self.concept_labels[concept_label1])
                                        e2.append(con2[3])
                                        e2.append(con2[4])
                                        e2.append(self.concept_labels[concept_label2])
                                        tokenized_sentence = self.split_tokens_2_sentence(l[0])
                                        dist_sentence_e1 = self.calc_distance(tokenized_sentence, e1)
                                        dist_sentence_e2 = self.calc_distance(tokenized_sentence, e2)

                                        include_data = 1
                                        # print (e1, e2)
                                        # include an example only if the distance is not too large!
                                        if abs(e1[1] - e2[0]) > self.max_word_dist:
                                            include_data = 0
                                            if r[0] != "NO_REL":
                                                print(str(abs(e1[1] - e2[0])), ":", filename,
                                                      self.split_tokens_2_sentence(l[0]), con1, e1, e2, r[0],
                                                      dist_sentence_e1, dist_sentence_e2)

                                        if include_data == 1:
                                            relation_sent_set.append(
                                                [filename, self.split_tokens_2_sentence(l[0]), e1, e2,
                                                 r[0] + rel_extend, dist_sentence_e1, dist_sentence_e2])
                                        # self.relation_sent_set.append([filename, l[0], l[1] - diff, l[2] - diff, con1, e1, con2, e2, r])

                for r in relations:
                    if r[3] == 0:
                        print("missed relation! >>", r)

        return relation_sent_set

    def get_word_pos(self, token_offsets, off_s, off_e):

        w_s = -1
        w_e = -1

        cnt = 0
        for tok in token_offsets:
            # print (tok, off_s, off_e)
            if tok[0] <= off_s and tok[1] >= off_s:
                # print ("1")
                w_s = cnt
            if tok[0] < off_e and tok[1] >= off_e:
                # print ("2")
                w_e = cnt + 1
            cnt += 1

        # fall-back
        if w_s == -1:
            w_s = w_e - 1
        if w_e == -1:
            w_e = w_s + 1

        return [w_s, w_e]

    def print_relations(self):
        for r in self.relation_sent_set:
            print(">", r)

    def calc_distance(self, tokenized_sentence, ent):
        dist_array = [None] * len(tokenized_sentence)
        dist = -(ent[0])

        for i in range(len(tokenized_sentence)):
            if dist == 0:
                dist_array[i] = dist
                if i == (ent[1] - 1):
                    dist += 1
            else:
                dist_array[i] = dist
                dist += 1
        return dist_array

    # used to parse metamap output
    def get_word_idx(self, wsplit, x_line):
        if int(wsplit[1]) > 2:
            tmp_str = x_line[:int(wsplit[0]) - 1]

            tmp_str = re.sub("  *", " ", tmp_str)
            tmpa = tmp_str.split(" ")

            words_before = len(tmpa) - 1
            tmp_result = []
            tmp_result.append(words_before)

            current_word = x_line[int(wsplit[0]) - 1:int(wsplit[0]) - 1 + int(wsplit[1])]

            tmpa_sz = len(current_word.split(" "))
            if tmpa_sz > 1:
                tmpa_sz -= 1
                for i in range(tmpa_sz):
                    tmp_result.append(words_before + (i + 1))

            return tmp_result
        return []

    def get_umls_info(self, xarray, outfile):

        file_object = open(outfile, "w")
        prev_line = ""
        cui_sentence = []

        for xi in range(len(xarray)):

            # print (">>>", xi)

            x_line = " ".join(xarray[xi][1])
            x_line = re.sub("\n", "", x_line)

            # init empfty array
            cui_sentence = x_line.split(" ")
            for c in range(len(cui_sentence)):
                cui_sentence[c] = "NULL"

            file_name = xarray[xi][0]
            text_line = x_line

            prev_line = x_line

            file_object.write(
                xarray[xi][0] + "|" + str(xarray[xi][2][3]) + "|" + str(xarray[xi][3][3]) + "|" + x_line + "|" + (
                    " ".join(cui_sentence)) + "\n")

        file_object.close()
        return xarray

    def add_concept_id_data(self, train, test):

        def get_all_pos(pos_list):
            location = []
            for i in range(len(pos_list)):
                if pos_list[i] == 0:
                    location.append(i)
            return location

        for i in range(len(train)):
            #print(train[i])
            sentence = train[i][1]
            entity_1 = train[i][2]
            entity_2 = train[i][3]
            #pos_1 = train[i][5].index(0)
            pos_1 = get_all_pos(train[i][5])
            #pos_2 = train[i][6].index(0)
            pos_2 = get_all_pos(train[i][6])
            #print(pos_1, pos_2)

            concept_id_1 = self.all_concepts_set[entity_1[2]]
            concept_id_2 = self.all_concepts_set[entity_2[2]]
            #print(concept_id_1)
            #print(train[i])
            #print(sentence, pos_1, pos_2, entity_1, entity_2, concept_id_1, concept_id_2)
            #print(">> POS_1", train[i][5], pos_1)
            #print(">> POS_2", train[i][6], pos_2)
            list_ids = []
            list_ids_2 = []

            for j in range(len(sentence)):
                if j in pos_1:
                    list_ids.append(concept_id_1[0])

                else:
                    list_ids.append(0)


            for j in range(len(sentence)):
                if j in pos_2:
                    list_ids_2.append(concept_id_2[0])
                else:
                    list_ids_2.append(0)


            #print("ID_List: ", list_ids)
            #print("List_Token_ID: ", list_ids_2, "\n")
            train[i].append(list_ids)
            train[i].append(list_ids_2)

        for i in range(len(test)):
            #print(test[i])
            sentence = test[i][1]
            entity_1 = test[i][2]
            entity_2 = test[i][3]
            #pos_1 = test[i][5].index(0)
            pos_1 = get_all_pos(test[i][5])
            #pos_2 = test[i][6].index(0)
            pos_2 = get_all_pos(test[i][6])
            #print(pos_1, pos_2)

            concept_id_1 = self.all_concepts_set[entity_1[2]]
            concept_id_2 = self.all_concepts_set[entity_2[2]]
            #print(concept_id_1)
            #print(test[i])
            #print(sentence, pos_1, pos_2, entity_1, entity_2, concept_id_1, concept_id_2)
            #print(">> POS_1", test[i][5], pos_1)
            #print(">> POS_2", test[i][6], pos_2)
            list_ids = []
            list_ids_2 = []

            for j in range(len(sentence)):
                if j in pos_1:
                    list_ids.append(concept_id_1[0])

                else:
                    list_ids.append(0)

            for j in range(len(sentence)):
                if j in pos_2:
                    list_ids_2.append(concept_id_2[0])
                else:
                    list_ids_2.append(0)

            # print("ID_List: ", list_ids)
            # print("List_Token_ID: ", list_ids_2, "\n")
            test[i].append(list_ids)
            test[i].append(list_ids_2)

        for i in train:
            print(i[1:])


    def preprocess(self, train_d, test_d):

        print("Building vocabulary...")
        vocabulary = Vocabulary(self.unknown_token, self.padding_token, self.embeddings_norm, self.embeddings_dim)

        for r in train_d:
            for token in r[1]:
                vocabulary.add_word(token)

        for r in test_d:
            for token in r[1]:
                vocabulary.add_word(token)

        print("Loading pre-trained word embeddings...", self.embeddings_file)
        # vocabulary.load_word2vec_embeddings(self.embeddings_file, "words")
        vocabulary.load_word2vec_embeddings(self.embeddings_file, "bin")  # "normal"
        vocabulary.print_stats()
        vocabulary.fill_embeddings(fill_randomly=True)
        if self.min_word_frequency > 1:
            print("Removing words less frequent than %i" % self.min_word_frequency)
            vocabulary.remove_rare_words(min_frequency=self.min_word_frequency)

        print("Writing vocabulary...")
        vocabulary.write(self.vocabulary_dir)

        #############################

        # print vector file
        line_cnt = 1
        line_cnt = self.print_file(self.out_file_train, line_cnt, train_d, vocabulary)
        line_cnt = self.print_file(self.out_file_test, line_cnt, test_d, vocabulary)

        print("line_cnt:", line_cnt)

    def split_tokens(self, in_text):
        word_tokenizer = nltk.tokenize.WordPunctTokenizer()
        token_offsets = word_tokenizer.span_tokenize(in_text)
        return [Token(token_offset, in_text) for token_offset in token_offsets]

    def split_tokens_2_sentence(self, in_text):
        word_tokenizer = nltk.tokenize.WordPunctTokenizer()
        token_offsets = word_tokenizer.span_tokenize(in_text)
        return [in_text[token_offset[0]:token_offset[1]] for token_offset in token_offsets]

    def print_file(self, out_file, l_cnt, rel_set, voc):
        with open(out_file, 'w') as output_file:
            for i in range(len(rel_set)):

                #print("=>>>", rel_set[i])
                tok_array = rel_set[i][1]
                word_indexes_str = " ".join([str(voc.get_word_index(token_text.strip())) for token_text in tok_array])

                rel_set[i].append(word_indexes_str)


                rel_label = rel_set[i][4]
                first_args_string = " ".join([str(v) for v in rel_set[i][5]])
                second_args_string = " ".join([str(v) for v in rel_set[i][6]])

                concept_ids_string = " ".join([str(v) for v in rel_set[i][7]])
                concept_ids_string_2 = " ".join([str(v) for v in rel_set[i][8]])
                print(concept_ids_string)
                # word_indexes_str = rel_set[i][7]

                if rel_label == "NO_REL(e1,e2)":
                    rel_label = "Other"

                if self.add_concept_embed == 1:
                    if self.ner_labels == 1:
                        output_file.write("%s:%s:%s:%s:%s:%s:%s:%s:%s\n" %
                                          (l_cnt, "rel-" + rel_label, word_indexes_str, first_args_string,
                                           second_args_string, rel_set[i][2][4], rel_set[i][3][4], concept_ids_string, concept_ids_string_2))
                        l_cnt += 1
                    else:
                        output_file.write("%s:%s:%s:%s:%s:%s:%s\n" %
                                          (l_cnt, "rel-" + rel_label, word_indexes_str, first_args_string,
                                           second_args_string, concept_ids_string, concept_ids_string_2))
                        l_cnt += 1

                else:

                    if self.ner_labels == 1:
                        output_file.write("%s:%s:%s:%s:%s:%s:%s\n" %
                                          (l_cnt, "rel-" + rel_label, word_indexes_str, first_args_string,
                                           second_args_string, rel_set[i][2][4], rel_set[i][3][4]))
                        l_cnt += 1
                    else:
                        output_file.write("%s:%s:%s:%s:%s\n" %
                                          (l_cnt, "rel-" + rel_label, word_indexes_str, first_args_string,
                                           second_args_string))
                        l_cnt += 1

        return l_cnt



def main():
    argparser = argparse.ArgumentParser(description='Preprocessor for data in the format of BRAT')
    argparser.add_argument('-c', '--config', help='path to a configuration file')
    args = argparser.parse_args()

    if args.config is not None:
        config_file = args.config
    else:
        config_file = 'configs/brat2vec_conf.cfg'

    config = configparser.ConfigParser()
    print("Reading config from " + config_file + "\n")
    config.read(config_file)
    brat2vec = Brat2Vec(config)



    train_data = brat2vec.read_brat(config['Files'].get('brat_train_dir'))
    test_data = brat2vec.read_brat(config['Files'].get('brat_test_dir'))

    brat2vec.add_concept_id_data(train_data, test_data)
    brat2vec.preprocess(train_data, test_data)



    brat2vec.print_relations()
    pickle.dump(brat2vec.all_concepts_set, open(brat2vec.preprocessor_dir+"ner_labels_map_to_int.pickle", "wb"))



if __name__ == "__main__":
    main()
