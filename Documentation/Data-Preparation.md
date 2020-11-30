# Data Preparation
## Usage:
We use different scripts to convert the BRAT data into Training data depending on the task type. 
---
### Tokenize & Recalculate BRAT Offsets:
This script takes in two sets of datasets with the same content but different offsets (e.g. Tokenized), then re-generates
BRAT data for the the new dataset.
#### Parameters:
 ```
[ARGUMENTS]
old_data_input_path = ../../Data/fc_brat_data/
output_path = ./Jcore_Tokenized_new_Offset_Brat/

# 0 XMI, 1 Spacy, -1 No Tokenization
xmi_or_spacy = 0
 ```
---
### Brat2Conll:
This script converts the BRAT data into CoNLL data which will be used to train the NER model. <br>
The **keep_or_remove_mode** with **remove_concepts** and **keep_concepts** are used to specify the entities that you want to keep or remove. <br>
The **priority_list** is used to resolve conflicts between entities that Span onto the same token (entity with higher *>* priority will be kept).
#### Parameters:
 ```
[ARGUMENTS]
input_folder = ../TokenizeRcalcOffsetBratData/Jcore_Tokenized_new_Offset_Brat/

output_folder = Clean_Conll_Data_For_NER_Train/

attach_label = True
attach_id = False
concept_mode = 1

merge_multi_con = 0

generate_new_ann = True

# 0 is to use the keep_concepts list and keep the concepts that are listed in the list, 1 is to use the remove_concept list and remove the concepts that are listed in the list
keep_or_remove_mode = 0

remove_concepts = Medical_condition,Kommentar,Temporal_course,Negation,Speculation,Type,Structure_element
keep_concepts = Medical_condition,Measurement,Body_part,Treatment,DiagLab_Procedure,State_of_health,Process,Medication,Time_information,Local_specification,Biological_chemistry,Biological_parameter,Dosing,Person,Medical_specification,Medical_device,Body_Fluid,Degree,Tissue
priority_list = Medical_condition>Measurement>Body_part>Treatment>DiagLab_Procedure>State_of_health>Process>Medication>Time_information>Local_specification>Biological_chemistry>Biological_parameter>Dosing>Person>Medical_specification>Medical_device>Body_Fluid>Degree>Tissue

 ```
---
### Brat2Vec:
This script converts the BRAT data into RelVec data which will be used to train the RelEx model.
#### Parameters:
 ```
[Files]
preprocessor_dir = outputs/1_fake_clinical_data_vec_format/

brat_train_dir = ../../Data/fc_brat_data/
# Dummy path will have no effect you could replace it if you want to generate test data directly
brat_test_dir = ../../Data/fake_clinical_data_Jcore_CONLL/

out_file_train = outputs/1_fake_clinical_data_vec_format/all_data.txt
out_file_test = outputs/1_fake_clinical_data_vec_format/test.txt

embeddings_file = resources/embedding_file/PubMed-shuffle-win-30.bin

[Preprocessor]
unknown_token = _UNK
padding_token = _PAD
min_word_frequency = 1
sentence_split = 1
include_ner_labels = 0
always_first_first_occurrence = 1
use_graphembed = 1
add_concept_embeddings = 1

[Model]
max_word_dist = 40
embeddings_dim = 200
graphembedding_dim = 300
 ```
---