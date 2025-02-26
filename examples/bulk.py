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
# ]
# ///

import marimo

__generated_with = "0.11.9"
app = marimo.App(width="medium", layout_file="layouts/bulk.grid.json")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from sentence_transformers import SentenceTransformer
    import altair as alt
    from sklearn.metrics.pairwise import cosine_similarity
    return SentenceTransformer, alt, cosine_similarity, mo, pl


@app.cell
def _(pl):
    df = pl.read_csv("examples/data/spam.csv", encoding="latin1").select(label=pl.col("v1"), text=pl.col("v2"))
    texts = df["text"].to_list()
    return df, texts


@app.cell
def _(SentenceTransformer, texts):
    tfm = SentenceTransformer("all-MiniLM-L6-v2")
    X = tfm.encode(texts)
    return X, tfm


@app.cell
def _(X):
    from umap import UMAP

    umap_tfm = UMAP()
    X_tfm = umap_tfm.fit_transform(X)
    return UMAP, X_tfm, umap_tfm


@app.cell
def _(text_input):
    text_input.value
    return


@app.cell
def _(X, X_tfm, cosine_similarity, pl, text_input, texts, tfm):
    df_emb = (
        pl.DataFrame({
            "x": X_tfm[:, 0], 
            "y": X_tfm[:, 1], 
            "index": range(X.shape[0]), 
            "text": texts
        }).with_columns(sim=pl.lit(1))
    )

    if text_input.value:
        query = tfm.encode([text_input.value])
        similarity = cosine_similarity(query, X)[0]
        df_emb = df_emb.with_columns(sim=similarity)
    return df_emb, query, similarity


@app.cell
def _(mo):
    btn_spam = mo.ui.button(label="Annotate spam")
    btn_ham = mo.ui.button(label="Annotate ham")
    btn_undo = mo.ui.button(label="Undo")
    return btn_ham, btn_spam, btn_undo


@app.cell
def _(btn_ham, btn_spam, btn_undo, mo):
    mo.hstack([btn_ham, btn_spam, btn_undo])
    return


@app.cell
def _(chart, get_label, set_label):
    def undo():
        current_labels = get_label()
        for val in chart.value["index"]: 
            new_spam = [i for i in current_labels["spam"] if i != val]
            new_ham = [i for i in current_labels["ham"] if i != val]
        set_label({"spam": new_spam, "ham": new_ham})
    return (undo,)


@app.cell
def _(chart, get_label, set_label):
    def add_label(lab):
        current_labels = get_label()
        if lab == "spam": 
            for val in chart.value["index"]: 
                new_spam = list(set(current_labels["spam"] + [i for i in chart.value["index"]]))
                new_ham = [i for i in current_labels["ham"] if i != val]
        if lab == "ham":
            for val in chart.value["index"]: 
                new_ham = list(set(current_labels["spam"] + [i for i in chart.value["index"]]))
                new_spam = [i for i in current_labels["ham"] if i != val]
            
        set_label({"spam": new_spam, "ham": new_ham})
    return (add_label,)


@app.cell
def _(alt):
    def scatter(df):
        return (alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X("x:Q"),
            y=alt.Y("y:Q"),
            color=alt.Opacity("sim:Q")
        ).properties(width=500, height=500))
    return (scatter,)


@app.cell
def _(mo):
    get_label, set_label = mo.state({"spam": [], "ham": []})
    return get_label, set_label


@app.cell
def _():
    return


@app.cell
def _(df_emb, mo, scatter):
    chart = mo.ui.altair_chart(scatter(df_emb))
    chart
    return (chart,)


@app.cell
def _(chart):
    chart.value["text"]
    return


@app.cell
def _(mo):
    text_input = mo.ui.text_area(label="Color reference")
    text_input
    return (text_input,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
