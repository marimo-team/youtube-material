# /// script
# requires-python = "==3.10"
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
# ]
# ///

import marimo

__generated_with = "0.11.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""### Bulk labelling demo""")
    return


@app.cell
def _(mo, pl):
    with mo.status.spinner(subtitle="Loading data ...") as _spinner:
        df = pl.read_csv("examples/data/spam.csv", encoding="latin1").select(label=pl.col("v1"), text=pl.col("v2"))
        texts = df["text"].to_list()
    return df, texts


@app.cell
def _(SentenceTransformer, mo, texts):
    with mo.status.spinner(subtitle="Creating embeddings ...") as _spinner:
        tfm = SentenceTransformer("all-MiniLM-L6-v2")
        X = tfm.encode(texts)
    return X, tfm


@app.cell
def _(X, mo):
    with mo.status.spinner(subtitle="Running UMAP ...") as _spinner:
        from umap import UMAP

        umap_tfm = UMAP()
        X_tfm = umap_tfm.fit_transform(X)
    return UMAP, X_tfm, umap_tfm


@app.cell
def _(add_label, mo, undo):
    btn_spam = mo.ui.button(label="Annotate spam", on_click=lambda d: add_label("spam"))
    btn_ham = mo.ui.button(label="Annotate ham", on_click=lambda d: add_label("ham"))
    btn_undo = mo.ui.button(label="Undo", on_click=lambda d: undo())
    return btn_ham, btn_spam, btn_undo


@app.cell
def _(br, btn_ham, btn_spam, btn_undo, chart, form, json_download, mo, switch):
    mo.vstack([
        mo.hstack([btn_ham, btn_spam, btn_undo, switch, json_download]),
        br(),
        form if switch.value else "", 
        br() if switch.value else "",
        chart
    ])
    return


@app.cell
def _(chart):
    chart.value["text"]
    return


@app.cell
def _(chart, get_label, set_label):
    def undo():
        current_labels = get_label()
        new_spam = set(current_labels["spam"]).difference(chart.value["index"])
        new_ham = set(current_labels["ham"]).difference(chart.value["index"])
        set_label({"spam": list(new_spam), "ham": list(new_ham)})
    return (undo,)


@app.cell
def _():
    from mohtml import br
    return (br,)


@app.cell
def _(chart, get_label, set_label):
    def add_label(lab):
        current_labels = get_label()
        if lab == "spam": 
            new_ham = list(set(current_labels["ham"]).difference(chart.value["index"]))
            new_spam = list(set(current_labels["spam"]).union(chart.value["index"]))
        if lab == "ham":
            new_ham = list(set(current_labels["ham"]).union(chart.value["index"]))
            new_spam = list(set(current_labels["spam"]).difference(chart.value["index"]))

        set_label({"spam": new_spam, "ham": new_ham})
    return (add_label,)


@app.cell
def _(mo):
    get_label, set_label = mo.state({"spam": [], "ham": []})
    return get_label, set_label


@app.cell
def _(mo):
    text_input = mo.ui.text_area(label="Reference sentences")
    form = mo.md("""{text_input}""").batch(text_input=text_input).form()
    return form, text_input


@app.cell
def _(df_emb, labels, mo):
    from collections import Counter

    with mo.status.spinner(subtitle="Starting UI ...") as _spinner:
        df_emb

    Counter(labels)
    return (Counter,)


@app.cell
def _(df_emb, mo, pl):
    import json

    data = df_emb.filter(pl.col("label") != "unlabeled").select("text", "label").to_dicts()

    json_download = mo.download(
        data=json.dumps(data).encode("utf-8"),
        filename="data.json",
        mimetype="application/json",
        label="Download JSON",
    )
    return data, json, json_download


@app.cell
def _():
    return


@app.cell
def _(df_emb, mo, scatter):
    chart = mo.ui.altair_chart(scatter(df_emb))
    return (chart,)


@app.cell
def _(mo):
    switch = mo.ui.switch(False, label="Use search")
    return (switch,)


@app.cell
def _(alt, switch):
    def scatter(df):
        return (alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X("x:Q"),
            y=alt.Y("y:Q"),
            color=alt.Color("sim:Q") if switch.value else alt.Color("label:N", scale=alt.Scale(
               domain=['unlabeled', 'ham', 'spam'],
               range=['steelblue', 'green', 'red']
            ))
        ).properties(width=500, height=500))
    return (scatter,)


@app.cell
def _():
    return


@app.cell
def _(X, X_tfm, cosine_similarity, form, get_label, np, pl, texts, tfm):
    df_emb = (
        pl.DataFrame({
            "x": X_tfm[:, 0], 
            "y": X_tfm[:, 1], 
            "index": range(X.shape[0]), 
            "text": texts
        }).with_columns(sim=pl.lit(1))
    )

    if form.value:
        query = tfm.encode([form.value["text_input"]])
        similarity = cosine_similarity(query, X)[0]
        df_emb = df_emb.with_columns(sim=similarity)

    spam = set(get_label()["spam"])
    ham = set(get_label()["ham"])

    labels = []
    for i in range(df_emb.shape[0]):
        if i in spam:
            labels.append("spam")
        elif i in ham:
            labels.append("ham")
        else:
            labels.append("unlabeled")

    df_emb = df_emb.with_columns(label=np.array(labels))
    return df_emb, ham, i, labels, query, similarity, spam


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from sentence_transformers import SentenceTransformer
    import altair as alt
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.linear_model import LogisticRegression
    return (
        LogisticRegression,
        SentenceTransformer,
        alt,
        cosine_similarity,
        mo,
        np,
        pl,
    )


if __name__ == "__main__":
    app.run()
