import spacy
from spacy.tokens import Span
import streamlit as st
from typing import Optional
from spacy import displacy
from spacy.displacy import DependencyRenderer, EntityRenderer
from spacy.tokens import Doc, Span
from spacy.errors import Errors, Warnings
import warnings
import re
from spacy_streamlit.util import get_svg
from Pipeline.pipeline_for_manual import MedicalIEPipeline
from spacy.displacy import parse_ents, parse_deps
from spacy_streamlit import visualize_ner


_html = {}
RENDER_WRAPPER = None


def visualize_parser_new(
    doc: spacy.tokens.Doc,
    *,
    title: Optional[str] = "Named-Entity-Recognition & Relation-Extraction",
    sidebar_title: Optional[str] = "Options",
    key: Optional[str] = None,
) -> None:
    """Visualizer for dependency parses."""
    if title:
        st.header(title)
    if sidebar_title:
        st.sidebar.header(sidebar_title)
    split_sents = st.sidebar.checkbox(
        "Split sentences", value=False, key=f"{key}_parser_split_sents"
    )
    options = {
        "compact": st.sidebar.checkbox("Compact mode", value=True, key=f"{key}_parser_compact"),
    }

    arcs = []
    for rel in doc._.rels:
        if rel[0][0].i < rel[1][0].i:
            arcs.append({"start": rel[0][0].i, "end": rel[1][0].i, "label": rel[2].split('-')[1],
                      "dir": "left"})

        elif rel[0][0].i > rel[1][0].i:
            arcs.append({
                "start": rel[0][0].i,
                "end": rel[1][0].i,
                "label": rel[2].split('-')[1],
                "dir": "right",
            })

    docs = [span.as_doc() for span in doc.sents] if split_sents else [doc]
    for sent in docs:
        html = displacy.render(sent, options=options, style="dep", arcs=arcs)
        # Double newlines seem to mess with the rendering
        html = html.replace("\n\n", "\n")
        if split_sents and len(docs) > 1:
            st.markdown(f"> {sent.text}")
        st.write(get_svg(html), unsafe_allow_html=True)

def render_new(
    docs, style="dep", page=False, minify=False, jupyter=None, options={}, manual=False, arcs=[]
):
    """Render displaCy visualisation.

    docs (list or Doc): Document(s) to visualise.
    style (unicode): Visualisation style, 'dep' or 'ent'.
    page (bool): Render markup as full HTML page.
    minify (bool): Minify HTML markup.
    jupyter (bool): Override Jupyter auto-detection.
    options (dict): Visualiser-specific options, e.g. colors.
    manual (bool): Don't parse `Doc` and instead expect a dict/list of dicts.
    RETURNS (unicode): Rendered HTML markup.

    DOCS: https://spacy.io/api/top-level#displacy.render
    USAGE: https://spacy.io/usage/visualizers
    """
    factories = {
        "dep": (DependencyRenderer, parse_rels),
        "ent": (EntityRenderer, parse_ents),
    }
    if style not in factories:
        raise ValueError(Errors.E087.format(style=style))
    if isinstance(docs, (Doc, Span, dict)):
        docs = [docs]
    docs = [obj if not isinstance(obj, Span) else obj.as_doc() for obj in docs]
    if not all(isinstance(obj, (Doc, Span, dict)) for obj in docs):
        raise ValueError(Errors.E096)
    renderer, converter = factories[style]
    renderer = renderer(options=options)
    parsed = [converter(doc, arcs, options) for doc in docs] if not manual else docs
    _html["parsed"] = renderer.render(parsed, page=page, minify=minify).strip()
    html = _html["parsed"]
    if RENDER_WRAPPER is not None:
        html = RENDER_WRAPPER(html)

    return html


def render_old(
    docs, style="dep", page=False, minify=False, jupyter=None, options={}, manual=False
):
    """Render displaCy visualisation.

    docs (list or Doc): Document(s) to visualise.
    style (unicode): Visualisation style, 'dep' or 'ent'.
    page (bool): Render markup as full HTML page.
    minify (bool): Minify HTML markup.
    jupyter (bool): Override Jupyter auto-detection.
    options (dict): Visualiser-specific options, e.g. colors.
    manual (bool): Don't parse `Doc` and instead expect a dict/list of dicts.
    RETURNS (unicode): Rendered HTML markup.

    DOCS: https://spacy.io/api/top-level#displacy.render
    USAGE: https://spacy.io/usage/visualizers
    """
    factories = {
        "dep": (DependencyRenderer, parse_deps),
        "ent": (EntityRenderer, parse_ents),
    }
    if style not in factories:
        raise ValueError(Errors.E087.format(style=style))
    if isinstance(docs, (Doc, Span, dict)):
        docs = [docs]
    docs = [obj if not isinstance(obj, Span) else obj.as_doc() for obj in docs]
    if not all(isinstance(obj, (Doc, Span, dict)) for obj in docs):
        raise ValueError(Errors.E096)
    renderer, converter = factories[style]
    renderer = renderer(options=options)
    parsed = [converter(doc, options) for doc in docs] if not manual else docs
    _html["parsed"] = renderer.render(parsed, page=page, minify=minify).strip()
    html = _html["parsed"]
    if RENDER_WRAPPER is not None:
        html = RENDER_WRAPPER(html)

    return html



def parse_rels(orig_doc, arcs, options={}):
    """Generate dependency parse in {'words': [], 'arcs': []} format.

    doc (Doc): Document do parse.
    RETURNS (dict): Generated dependency parse keyed by words and arcs.
    """
    doc = Doc(orig_doc.vocab).from_bytes(orig_doc.to_bytes(exclude=["user_data"]))
    if not doc.is_parsed:
        warnings.warn(Warnings.W005)
    words = [
        {
            "text": w.text,
            "tag": w.ent_type_,
            "lemma": None,
        }
        for w in doc
    ]

    return {"words": words, "arcs": arcs, "settings": {'lang': 'de', 'direction': 'ltr'}}

def normalize_text(text):
    return re.sub( '\\s+', ' ', text).strip()



input = "Wegen Hypocalciaemie stationaer auf der 134 vom 22.7. bis 3.8.04 ."

med_pipeline = MedicalIEPipeline()

st.title('Visualization App:')
option = st.selectbox('Mode:', ['NER Visualize', 'NER & RelEx Visualize'], index=0)
sentence = st.text_input('Enter Sentence:', value=input)
text = normalize_text(sentence)
doc = med_pipeline.get_annotated_document(text)

if option == 'NER Visualize':
    spacy.displacy.render = render_old

    ents = list(doc.ents)

    for i in range(len(ents)):
        old_ent = ents[i]
        new_ent = Span(doc, old_ent.start, old_ent.end,
                       label=''.join(list(map(lambda x: x.upper(), old_ent.label_.split('_')))))
        ents[i] = new_ent

    doc.ents = ents
    tags = ['Medical_condition', 'Measurement', 'Body_part', 'Treatment', 'DiagLab_Procedure', 'State_of_health',
            'Process', 'Medication', 'Time_information', 'Local_specification', 'Biological_chemistry',
            'Biological_parameter', 'Dosing', 'Person', 'Medical_specification', 'Medical_device', 'Body_Fluid',
            'Degree', 'Tissue']
    colors = ['#E8DAEF', '#85C1E9', '#FAD7A0', '#ABEBC6', '#F7DC6F', '#F9E79F', '#A9DFBF', '#7FB3D5', '#F5B041',
              '#AED6F1', '#82E0AA', '#F4D03F', '#58D68D', '#A2D9CE', '#F8C471', '#D2B4DE', '#D7BDE2', '#76D7C4',
              '#614ec2', '#f59b47']
    tags = tuple(list(map(lambda x: ''.join(list(map(lambda y: y.upper(), x.split('_')))), tags)))
    col_dict = {}
    for i in range(len(tags)):
        col_dict[tags[i]] = colors[i]

    visualize_ner(doc, labels=tags, colors=col_dict)

else:
    spacy.displacy.render = render_new

    visualize_parser_new(doc)

