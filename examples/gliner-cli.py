# /// script
# requires-python = "==3.12"
# dependencies = [
#     "gliner==0.2.13",
#     "marimo",
#     "spacy==3.8.3",
#     "srsly==2.5.1",
# ]
# ///

import marimo

__generated_with = "0.11.6"
app = marimo.App(width="medium")


@app.cell
def _(labels_ui, mo, text_ui):
    markdown = mo.md(
        '''
        ## Marimo/GliNER demo

        This notebook is a bit special. It defines a UI *and* a CLI in one single file!

        - What is the input text?: {text}
        - What named entities would you like to detect?: {labels}
        '''
    )

    batch = mo.ui.batch(
        markdown, {"text": text_ui, "labels": labels_ui}
    ).form()

    batch
    return batch, markdown


@app.cell
def _(batch, displacy, mo, model, spacy):
    nlp = spacy.blank("en")

    def text_to_doc(text, labels):
        labels = labels.split(",")
        entities = model.predict_entities(text, labels)

        doc = nlp(text)
        doc.ents = [doc.char_span(e['start'], e['end'], label=e['label']) for e in entities]
        return doc


    doc = text_to_doc(
        text=batch.value.get("text") if batch.value else "Hi there. My name is Vincent and I am a Patriots fan.", 
        labels=batch.value.get("labels") if batch.value else "person,sports-team"
    )
    mo.Html(displacy.render(doc, style="ent"))
    return doc, nlp, text_to_doc


@app.cell
def _(mo):
    text_ui = mo.ui.text_area(placeholder="Hi there. My name is Vincent and I am a Patriots fan.")
    labels_ui = mo.ui.text(placeholder="person,sports-team")
    return labels_ui, text_ui


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import spacy 
    from spacy import displacy
    from spacy.tokens import Span
    return Span, displacy, spacy


@app.cell
def _():
    from gliner import GLiNER

    model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
    return GLiNER, model


@app.cell
def _(mo, text_to_doc):
    import srsly

    if mo.app_meta().mode == "script": 
        args = mo.cli_args()
        stream_in = srsly.read_jsonl(args["path-in"])
        stream_out = (text_to_doc(d["text"], labels=args["labels"]).to_json() 
                      for d in stream_in)
        srsly.write_jsonl(args["path-out"], stream_out)
    return args, srsly, stream_in, stream_out


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
