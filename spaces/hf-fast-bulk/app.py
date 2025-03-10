# /// script
# requires-python = "==3.12"
# dependencies = [
#     "marimo",
#     "polars==1.23.0",
#     "sentence-transformers==3.4.1",
#     "umap-learn==0.5.7",
#     "llvmlite==0.44.0",
#     "altair==5.5.0",
#     "scikit-learn==1.6.1",
#     "numpy==2.1.3",
#     "mohtml==0.1.2",
#     "model2vec==0.4.0",
# ]
# ///

import marimo

__generated_with = "0.11.9"
app = marimo.App()


@app.cell
def _(mo):
    mo.md("""### Bulk labelling demo""")
    return


@app.cell
def _(mo, use_default_switch):
    uploaded_file = mo.ui.file(kind="area") if not use_default_switch.value else None
    uploaded_file
    return (uploaded_file,)


@app.cell
def _(mo):
    use_default_switch = mo.ui.switch(False, label="Use default dataset")
    use_default_switch
    return (use_default_switch,)


@app.cell
def _(mo):
    pos_label = mo.ui.text("pos", placeholder="positive label name")
    neg_label = mo.ui.text("neg", placeholder="negative label name")
    return neg_label, pos_label


@app.cell
def _(uploaded_file, use_default_switch):
    should_stop = not use_default_switch.value and len(uploaded_file.value) == 0
    return (should_stop,)


@app.cell
def _(mo, pl, should_stop, uploaded_file, use_default_switch):
    mo.stop(should_stop , mo.md("**Submit a dataset or use default one to continue.**"))

    if use_default_switch.value:
        df = pl.read_csv("spam.csv")
    else:
        df = pl.read_csv(uploaded_file.value[0].contents)

    texts = df["text"].to_list()
    return df, texts


@app.cell
def _(StaticModel, mo):
    with mo.status.spinner(subtitle="Loading model ...") as _spinner:
        tfm = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    return (tfm,)


@app.cell
def _(mo, texts, tfm):
    with mo.status.spinner(subtitle="Creating embeddings ...") as _spinner:
        X = tfm.encode(texts)
    return (X,)


@app.cell
def _(add_label, get_example, mo, neg_label, pos_label):
    btn_spam = mo.ui.button(
        label=f"Annotate {neg_label.value}", 
        on_click=lambda d: add_label(get_example(), neg_label.value), 
        keyboard_shortcut="Ctrl-L"
    )
    btn_ham = mo.ui.button(
        label=f"Annotate {pos_label.value}", 
        on_click=lambda d: add_label(get_example(), pos_label.value),
        keyboard_shortcut="Ctrl-K"
    )
    return btn_ham, btn_spam


@app.cell
def _(gen, get_label, set_example, set_label):
    def add_label(text, lab):
        current_labels = get_label()
        set_label(current_labels + [{"text": text, "label": lab}])
        set_example(next(gen))
    return (add_label,)


@app.cell
def _():
    from mohtml import br
    return (br,)


@app.cell
def _(mo):
    get_label, set_label = mo.state([])
    return get_label, set_label


@app.cell
def _(gen, mo):
    get_example, set_example = mo.state(next(gen))
    return get_example, set_example


@app.cell
def _(div, get_example, p):
    div(
        p(get_example()), 
        klass="bg-gray-100 p-4 rounded-lg"
    )
    return


@app.cell
def _(btn_ham, btn_spam, mo):
    mo.hstack([
        btn_ham, btn_spam
    ])
    return


@app.cell
def _():
    from mohtml import tailwind_css, div, p

    tailwind_css()
    return div, p, tailwind_css


@app.cell
def _(mo, should_stop):
    mo.stop(should_stop)

    text_input = mo.ui.text_area("Query can go here", label="Reference sentences")
    form = mo.md("""{text_input}""").batch(text_input=text_input).form()
    form
    return form, text_input


@app.cell
def _(get_label, mo):
    import json

    data = get_label()

    json_download = mo.download(
        data=json.dumps(data).encode("utf-8"),
        filename="data.json",
        mimetype="application/json",
        label="Download JSON",
    )
    return data, json, json_download


@app.cell
def _(X, cosine_similarity, form, mo, pl, texts, tfm):
    mo.stop(not form.value.get("text_input", None), "Need a text input to fetch example")

    df_emb = (
        pl.DataFrame({
            "index": range(X.shape[0]), 
            "text": texts
        }).with_columns(sim=pl.lit(1))
    )


    query = tfm.encode([form.value["text_input"]])
    similarity = cosine_similarity(query, X)[0]
    df_emb = df_emb.with_columns(sim=similarity).sort(pl.col("sim"), descending=True)
    gen = (_["text"] for _ in df_emb.head(100).to_dicts())
    return df_emb, gen, query, similarity


@app.cell
def _(get_label, mo, pl, should_stop):
    mo.stop(should_stop)

    pl.DataFrame(get_label())
    return


@app.cell
def _(mo):
    with mo.status.spinner(subtitle="Loading libraries ...") as _spinner:
        import polars as pl
        import altair as alt
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.linear_model import LogisticRegression
        from sklearn.decomposition import PCA
    return LogisticRegression, PCA, alt, cosine_similarity, np, pl


@app.cell
def _(mo):
    with mo.status.spinner(subtitle="Loading model2vec ...") as _spinner:
        from model2vec import StaticModel
    return (StaticModel,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
