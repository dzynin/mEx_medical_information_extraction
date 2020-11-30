from Pipeline.pipeline import MedicalIEPipeline
import re
from utils.document_helper import doc_to_brat
import os

def normalize_text(text):
    return re.sub( '\\s+', ' ', text).strip()


input = """Wegen Hypocalciaemie stationaer auf der 134 vom 22.7. bis 3.8.04 .
Gutes Befinden , keine Kraempfe , keine GI-Symptome , kein epigastrischen Schmerzen , Stuhlgang normal .
Letztes S-Calcium 1,94 unter orlaer Therapie mit Ca und Rocaltrol ."""



med_pipeline = MedicalIEPipeline()

data_dir = '../Data/fake_clinical_data/'
output_dir = '../Data/fc_brat_data/'
output_dir = '../Data/test/'

for file in os.listdir(data_dir):
    curr_txt = open(data_dir+file, encoding='utf8')
    text = normalize_text(curr_txt.read())
    doc = med_pipeline.get_annotated_document(text)
    brat = doc_to_brat(doc,
                       selected_ents=None,
                       selected_rels=None,
                       enable_negation=False,
                       enable_candidate_search=False,
                       enable_wsd=False)

    new_txt = open(output_dir+file, 'w+')
    new_txt.write(brat['text'])
    new_txt.close()

    new_brat = open(output_dir+file.replace('.txt', '.ann'), 'w+')

    tmp = []
    for ent in brat['entities']:
        tmp.append(ent[0]+'\t'+ent[1][:1]+ent[1].lower()[1:]+' '+str(ent[2][0][0])+' '+str(ent[2][0][1])+'\t'+ent[3])
    for rel in brat['relations']:
        tmp.append(rel[0]+'\t'+rel[1].split('-')[1].lower()+' arg1:'+rel[2][0][1]+' arg2:'+rel[2][1][1])
    tmp = '\n'.join(tmp)
    new_brat.write(tmp)
    new_brat.close()
    curr_txt.close()

