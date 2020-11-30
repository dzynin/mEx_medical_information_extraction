import os
import re
import configparser


config = configparser.ConfigParser()
config.read('Brat2Conll_config_file.ini')


class Brat2Conll:
    def __init__(self):
        self.input_folder = config['ARGUMENTS']['input_folder']
        self.output_folder = config['ARGUMENTS']['output_folder']

        if self.input_folder[-1] != "/":
            self.input_folder += "/"

        if self.output_folder[-1] != "/":
            self.output_folder += "/"

        self.merge_multi_con = int(config['ARGUMENTS']['merge_multi_con'])

        self.attach_label = config['ARGUMENTS']['attach_label']
        self.generate_new_ann = config['ARGUMENTS']['generate_new_ann']
        self.keep_or_remove_mode = int(config['ARGUMENTS']['keep_or_remove_mode'])
        self.remove_concepts = []
        self.keep_concepts = []
        self.priority_dictionary = {}

        # Parse the keep_concept and remove_concept lists + parse the priority_dictionary list
        if self.keep_or_remove_mode == 0:
            if config['ARGUMENTS']['keep_concepts'].split(",") == ['']:
                pass
            else:
                for i in config['ARGUMENTS']['keep_concepts'].split(","):
                    self.keep_concepts.append(i.replace(" ", ""))

        elif self.keep_or_remove_mode == 1:

            if config['ARGUMENTS']['remove_concepts'].split(",") == ['']:
                pass
            else:
                for i in config['ARGUMENTS']['remove_concepts'].split(","):
                    self.remove_concepts.append(i.replace(" ", ""))
        else:
            raise Exception

        if config['ARGUMENTS']['priority_list'].split(">") == ['']:
            pass
        else:
            counter = 0
            for i in config['ARGUMENTS']['priority_list'].split(">"):
                if i not in self.priority_dictionary:
                    self.priority_dictionary[i] = counter
                counter += 1

    # Follow the remove/keep entities algorithm
    def brat2conll(self):
        # If the output folder does not exist create a new one
        if not os.path.isdir(self.output_folder):
            os.mkdir(self.output_folder)

        # Get the names of the files combinations that exists in all the needed formats
        file_list = FileNamesGet().get_valid_names(self.input_folder)

        logger = Logger(file_list)

        # Iterate through the list of files
        for file_name in file_list:

            print(file_name)
            # Change to the input folder
            #os.chdir(self.input_folder)

            # Open the current text file
            curr_txt_file = open(self.input_folder + file_name + ".txt", encoding='utf8')

            # Open the current brat file
            curr_ann_file = open(self.input_folder + file_name + ".ann", encoding='utf8')

            # Read the content of the text file
            currText = Text(curr_txt_file.read())

            # If the text files ends with a new line remove it from the list of sentences
            if not currText.text_str.split("\n")[-1]:
                sentences_array = currText.text_str.split("\n")[0:-1]

            else:
                sentences_array = currText.text_str.split("\n")

            # Strip each sentence and create a list of senetence objects
            for i in range(len(sentences_array)):
                sentences_array[i] = Sentence(sentences_array[i].strip(), sentences_array[i].strip().split(" "))

            currText.sentence_array = sentences_array

            # calculate and update sentences/tokens offsets
            for i in range(len(currText.sentence_array)):
                if i == 0:
                    currText.sentence_array[i].start_offset = 0
                    currText.sentence_array[i].end_offset = len(currText.sentence_array[i].sentence_str)
                    currText.sentence_array[i].create_intern_tokens()
                else:
                    currText.sentence_array[i].start_offset = currText.sentence_array[i - 1].end_offset + 1
                    currText.sentence_array[i].end_offset = currText.sentence_array[i].start_offset + len(
                        currText.sentence_array[i].sentence_str)
                    currText.sentence_array[i].create_intern_tokens()

            # Go through brat file
            for x_line in curr_ann_file:
                x_line = re.sub("\n", "", x_line)
                if re.match("^T[0-9]", x_line):
                    cons = self.create_tconcept(x_line.split("\t"), currText)
                    for con in cons:
                        id = con.id.split("_")[0]
                        label = con.label

                        if self.keep_or_remove_mode == 0:
                            if label in self.keep_concepts:
                                currText.t_objects[id] = con
                                currText.t_obj_list.append(con)
                                logger.file_loggers[file_name].keep_concepts.append(con)
                            else:
                                logger.file_loggers[file_name].remove_concepts.append(con)

                        elif self.keep_or_remove_mode == 1:

                            if label not in self.remove_concepts:
                                # Append the new object to the whole list of TConceptObj's
                                currText.t_objects[id] = con
                                currText.t_obj_list.append(con)
                                logger.file_loggers[file_name].keep_concepts.append(con)
                            else:
                                logger.file_loggers[file_name].remove_concepts.append(con)

            currText.prio = self.priority_dictionary
            currText.t_obj_list = sorted(currText.t_obj_list, key=lambda x: int(x.offset_s))
            currText.filter_concepts(self.priority_dictionary, self.create_tconcept)
            currText.clean_t_obj_list()
            currText.map_sentence_concept()
            currText.build_conll_structure()
            currText.create_simple_conll()
            currText.convert_simple_conll_bio()
            #currText.print_simple_conll()
            currText.write_conll_file(file_name, self.output_folder)

    def create_tconcept(self, vals, textobj):
        offsets = list(map(lambda x: x.split(","), ",".join(vals[1].split(" ")[1:]).split(";")))
        for offset in offsets:
            for sentence in textobj.sentence_array:
                if sentence.start_offset <= int(offset[0]) and int(offset[1]) <= sentence.end_offset:
                    offset.append(sentence)
                    break
        try:
            groupby = {}
            for i in offsets:
                if i[2] not in groupby.keys():
                    groupby[i[2]] = [i[0]+","+i[1]]
                else:
                    groupby[i[2]] = groupby[i[2]] + [i[0] + "," + i[1]]
        except IndexError:
            pass

        res_cons = []
        index = 0
        for con in groupby.keys():
            t_obj = TConceptObj(vals[0]+"_"+str(index), vals[1].split(" ")[0], int(groupby[con][0].split(",")[0]), int(groupby[con][-1].split(",")[1]), textobj.text_str[int(groupby[con][0].split(",")[0]):int(groupby[con][-1].split(",")[1])], groupby[con])
            t_obj.create_substrs(textobj.text_str)
            res_cons.append(t_obj)
            index += 1

        return res_cons

class TConceptObj(object):
    def __init__(self, id, label, offset_s, offset_e, string, offset_list):
        self.id = id
        self.label = label
        self.offset_s = offset_s
        self.offset_e = offset_e
        self.string = string
        self.keep = True
        self.offset_list = offset_list
        self.sub_strings = []
        self.multi = False

    def create_substrs(self, text_str):
        for offs in self.offset_list:
            s = int(offs.split(",")[0])
            e = int(offs.split(",")[1])
            for k in text_str[s:e].split(" "):
                self.sub_strings.append(k)

        if len(self.sub_strings) > 1:
            self.multi = True


class Text(object):
    def __init__(self, text):
        self.text_str = text
        self.sentence_array = []
        self.t_objects = {}
        self.t_obj_list = []
        self.conll_list = []
        self.prio = {}
        self.simple_conll_list = []

    def write_conll_file(self, filename, out_dir):
        obi_file = open(out_dir+filename + ".conll", "w+")
        for i in range(len(self.simple_conll_list)):
            if not self.simple_conll_list[i]:
                obi_file.write('\n')
            else:
                if self.simple_conll_list[i] == len(self.simple_conll_list)-1:
                    obi_file.write("\t".join(self.simple_conll_list[i]) if self.simple_conll_list[i][1] != '' else self.simple_conll_list[i][0]+"\tO")
                else:
                    obi_file.write("\t".join(self.simple_conll_list[i])+"\n" if self.simple_conll_list[i][1] != '' else self.simple_conll_list[i][0]+"\tO\n")


    def filter_concepts(self, prio_list, create_func):
        while (self.overlap_indicator()):
            temp_new_merged_cons = []
            for i in range(len(self.t_obj_list)):
                if self.t_obj_list[i].keep:
                    for j in range(len(self.t_obj_list[i + 1:])):
                        token1 = self.t_obj_list[i]
                        token2 = self.t_obj_list[i + 1:][j]
                        if token1.keep and token2.keep:
                            # check for general type of overlap
                            if token1.offset_s <= token2.offset_e and token2.offset_s <= token1.offset_e:
                                # if the t_objs share the same label merge them into one
                                if token1.label == token2.label:
                                    start_offset = min(token1.offset_s, token2.offset_s)
                                    end_offset = max(token1.offset_e, token2.offset_e)
                                    new_str = self.text_str[start_offset:end_offset]
                                    res_con = self.create_simple_conll()
                                    cons = create_func([token1.id, token1.label+" "+str(start_offset)+" "+str(end_offset), new_str], self)

                                    temp_new_merged_cons.extend(cons)
                                    #.append(TConceptObj(token1.id, token1.label, start_offset, end_offset, new_str))

                                    token1.keep = False
                                    token2.keep = False
                                # choose one of them after different guidelines
                                else:
                                    token1_prio = prio_list[token1.label]
                                    token2_prio = prio_list[token2.label]

                                    # Exact same offsets: prio_list
                                    if token1.offset_s == token2.offset_s and token1.offset_e == token2.offset_e:
                                        if token1_prio < token2_prio:
                                            token2.keep = False
                                        if token1_prio > token2_prio:
                                            token1.keep = False
                                    # Token2 is nested in Token1: remove nested token2
                                    elif token1.offset_s < token2.offset_s and token2.offset_e < token1.offset_e:
                                        token2.keep = False
                                    # Token1 is nested in Token2: remove nested token1
                                    elif token2.offset_s < token1.offset_s and token1.offset_e < token2.offset_e:
                                        token1.keep = False
                                    # other type of overlap: prio_list
                                    else:
                                        if token1_prio < token2_prio:
                                            token2.keep = False
                                        if token1_prio > token2_prio:
                                            token1.keep = False

            self.t_obj_list.extend(temp_new_merged_cons)
            self.t_obj_list = sorted(self.t_obj_list, key=lambda x: int(x.offset_s))

    def overlap_indicator(self):
        for i in range(len(self.t_obj_list)):
            if self.t_obj_list[i].keep:
                for j in range(len(self.t_obj_list[i + 1:])):
                    token1 = self.t_obj_list[i]
                    token2 = self.t_obj_list[i + 1:][j]
                    if token2.keep:
                        # check for general type of overlap
                        if token1.offset_s <= token2.offset_e and token2.offset_s <= token1.offset_e:
                            return True
        return False

    def clean_t_obj_list(self):
        self.t_obj_list = list(filter(lambda a: a.keep, self.t_obj_list))

    def build_conll_structure(self):
        for sentence in self.sentence_array:
            for token in sentence.token_array:
                cons = self.map_token_concept(token, sentence)
                self.conll_list.append([token, cons])
            self.conll_list.append([])
        self.conll_list = self.conll_list[:-1]

    def map_token_concept(self, token, sentence):
        res = []
        for concept in sentence.concepts:
            if concept.keep:
                if token.start_offset <= concept.offset_e and concept.offset_s <= token.end_offset:
                    if concept.string in token.token_str or token.token_str in concept.string or token.token_str in concept.sub_strings or len([i for i in concept.sub_strings if i in token.token_str or token.token_str in i]) > 0:
                        res.append(concept)
        if len(res) > 1:
            return self.select_highest_prio(res)
        elif len(res) == 1:
            return res[0]
        else:
            return ""

    def map_sentence_concept(self):
        for sentence in self.sentence_array:
            for concept in self.t_obj_list:
                if sentence.start_offset <= concept.offset_s and concept.offset_e <= sentence.end_offset:
                    sentence.concepts.append(concept)

    def select_highest_prio(self, res_list):
        winner_index = -1
        min_prio = len(self.prio.keys()) + 10
        for i in range(len(res_list)):
            if self.prio[res_list[i].label] < min_prio:
                min_prio = self.prio[res_list[i].label]
                winner_index = i

        return res_list[winner_index]

    def create_simple_conll(self):
        for i in self.conll_list:
            if i != []:
                if i[1] != '':
                    self.simple_conll_list.append([i[0].token_str, i[1].label])
                else:
                    self.simple_conll_list.append([i[0].token_str, i[1]])
            else:
                self.simple_conll_list.append([])
        self.simple_conll_list = [i for i in self.simple_conll_list if i != ["", ""]]
        self.simple_conll_list = [i for i in self.simple_conll_list if i != ['\ufeff', ""]]

        try:
            while(self.simple_conll_list[0] == []):
                self.simple_conll_list.pop(0)
        except Exception:
            pass

        try:
            while(self.simple_conll_list[-1] == []):
                self.simple_conll_list.pop()
        except Exception:
            pass

        temp_list = []
        for i in range(len(self.simple_conll_list)):
            if self.simple_conll_list[i] == []:
                try:
                    if self.simple_conll_list[i+1] == []:
                        pass
                    else:
                        temp_list.append(self.simple_conll_list[i])
                except Exception:
                    pass
            else:
                temp_list.append(self.simple_conll_list[i])
        self.simple_conll_list = temp_list

    def convert_simple_conll_bio(self):

        for i in range(len(self.simple_conll_list)):
            x_line = self.simple_conll_list[i]
            if x_line != [] and x_line[1] != '':
                if x_line[1].split("-")[0] not in ["B", "I", "O"]:
                    fix_curr_con = x_line[1]
                    self.simple_conll_list[i][1] = "B-" + self.simple_conll_list[i][1]
                    j = i + 1
                    try:
                        while self.simple_conll_list[j][1] == fix_curr_con:
                            self.simple_conll_list[j][1] = "I-" + self.simple_conll_list[j][1]
                            j += 1
                    except IndexError:
                        pass

    def print_simple_conll(self):
        for x_line in self.simple_conll_list:
            print(x_line)

class Sentence(object):
    def __init__(self, sentence_str, tokens):
        self.sentence_str = sentence_str
        self.start_offset = -1
        self.end_offset = -1
        self.token_array = tokens
        self.concepts = []

    def create_intern_tokens(self):
        index = self.start_offset
        for i in range(len(self.token_array)):
            self.token_array[i] = Token(self.token_array[i], index, index + len(self.token_array[i]))
            index += len(self.token_array[i].token_str) + 1


class Token(object):
    def __init__(self, token_str, offset_s, offset_e):
        self.token_str = token_str
        self.start_offset = offset_s
        self.end_offset = offset_e
        self.labels = []


class Logger(object):
    def __init__(self, filenames):
        self.file_loggers = {}
        for name in filenames:
            self.file_loggers[name] = FileLogger(name)


class FileLogger(object):
    def __init__(self, filename):
        self.filename = filename
        self.keep_concepts = []
        self.remove_concepts = []
        self.sub_concepts = []


class FileNamesGet:
    # Set up needed information
    def __init__(self):
        self.input_folder = ""

    # Main extraction process
    def get_valid_names(self, input_folder):
        self.input_folder = input_folder

        # Get all brat file names
        #os.chdir(input_folder)
        file_names = os.listdir(input_folder)

        # Remove the ending from the files
        file_names_no_extension = []
        for i in range(len(file_names)):

            if file_names[i].endswith(".ann"):
                file_names_no_extension.append(file_names[i].replace('.ann', ''))

        # Iterate through the names and check if the file
        # is available in other formats, if so save the
        # file name
        valid_file_names = []
        for file in file_names_no_extension:
            # Check if the same file is in text format available
            text_extension = file + ".txt"
            text_file_check = os.path.isfile(input_folder+text_extension)

            if text_file_check:
                valid_file_names.append(file)
        return valid_file_names


def main():
    obj = Brat2Conll()
    obj.brat2conll()


if __name__ == '__main__':
    main()
