# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==5.5.0",
#     "anthropic==0.51.0",
#     "anywidget==0.9.18",
#     "chonkie[model2vec]==1.0.7",
#     "chromadb==1.0.9",
#     "diskcache==5.6.3",
#     "einops==0.8.1",
#     "marimo",
#     "model2vec==0.5.0",
#     "mohtml==0.1.10",
#     "mopad==0.4.0",
#     "numpy==2.2.6",
#     "pandas==2.2.3",
#     "polars==1.29.0",
#     "scikit-learn==1.6.1",
#     "scipy==1.15.3",
#     "sentence-transformers==4.1.0",
#     "srsly==2.5.1",
#     "traitlets==5.14.3",
# ]
# ///

import marimo

__generated_with = "0.13.11"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Setup of cache/db

    Boilerplate and loading of models.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import chromadb
    from diskcache import Cache

    client = chromadb.PersistentClient(path="my-chroma-db/")
    cache = {}
    return cache, client


@app.cell
def _():
    from sentence_transformers import SentenceTransformer
    from model2vec import StaticModel

    model_vec = StaticModel.from_pretrained("minishlab/potion-base-8M")
    model_nomic = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    models = {"model_vec": model_vec, "model_nomic": model_nomic}
    return (models,)


@app.cell
def _(cache):
    import srsly
    import hashlib
    from pathlib import Path


    def md5_hash(text):
        # Convert to bytes if it's a string
        if isinstance(text, str):
            text = text.encode("utf-8")

        # Create MD5 hash
        md5_hasher = hashlib.md5()
        md5_hasher.update(text)

        # Return the hexadecimal digest
        return md5_hasher.hexdigest()


    def file_to_items(path, model, model_name):
        urls = []
        for item in srsly.read_jsonl(path):
            embs = model.encode(item["sentences"])
            if (model_name, item["url"]) in cache:
                continue
            cache[(model_name, item["url"])] = item
            for sent, emb in zip(item["sentences"], embs):
                yield md5_hash(sent), sent, emb, {"url": item["url"]}
    return Path, file_to_items


@app.cell
def _(client, models):
    from datetime import datetime

    collections = {}
    for _name, _ in models.items():
        collections[_name] = client.get_or_create_collection(
            name=f"arxiv-{_name}",
            metadata={
                "description": f"Searching in the arxiv-frontpage app with {_name}",
                "created": str(datetime.now()),
            },
        )
    return (collections,)


@app.cell
def _(cache):
    len(cache)
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Adding documents

    Adding documents to our collections. One uses Model2Vec while the other uses Nomic embeddings.
    """
    )
    return


@app.cell
def _(mo):
    emb_btn = mo.ui.run_button(label="embed moar data")
    emb_btn
    return (emb_btn,)


@app.cell
def _(Path, collections, emb_btn, file_to_items, mo, models):
    from chromadb.errors import DuplicateIDError

    mo.stop(not emb_btn.value)

    file_generator = Path("../arxiv-frontpage/data/downloads").glob("*.jsonl")

    for _ in mo.status.progress_bar(
        range(5),
        title="Adding data to ChromaDB",
        subtitle="Working!",
        show_eta=True,
        show_rate=True,
    ):
        # Grab the next file
        file = next(file_generator)
        for mod_name, mod in models.items():
            print(mod_name, file)
            # Process each file's items in batches to ensure consistent lengths
            items = list(file_to_items(file, model=mod, model_name=mod_name))

            # If there are items to process, there are edge cases because generator
            if items:
                # Unzip the items into separate lists
                ids, docs, embs, metadatas = zip(*items)

                # Verify all lists have the same length before upserting
                assert len(ids) == len(docs) == len(embs) == len(metadatas), (
                    "Mismatched lengths in data"
                )

                try:
                    collections[mod_name].upsert(
                        documents=list(docs),
                        embeddings=list(embs),
                        metadatas=list(metadatas),
                        ids=list(ids),
                    )
                except DuplicateIDError:
                    print(f"skipping {file}, has duplicates")

            print({n: c.count() for n, c in collections.items()})
    return


@app.cell
def _(collections):
    {n: c.count() for n, c in collections.items()}
    return


@app.cell
def _(models):
    models
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Annotation code

    Boilerplate for annotation
    """
    )
    return


@app.cell
def _(get_annot):
    import polars as pl

    pl.DataFrame(get_annot())
    return (pl,)


@app.cell
def _(collections, div, mo, models, p, query_ui):
    mo.stop(query_ui.value is None)

    import random


    class AnnotationQueue:
        def __init__(self, query, n_results=10):
            self.query = query
            self.results = {
                n: c.query(
                    models[n].encode(self.query),
                    n_results=n_results,
                    include=["embeddings", "metadatas", "documents"],
                )
                for n, c in collections.items()
            }
            self.n_results = n_results

        def make_stream(self):
            self.current_index = 0

            for k in range(self.n_results):
                for n, res in self.results.items():
                    url = res["metadatas"][0][k]["url"]
                    doc = res["documents"][0][k]
                    yield {"name": n, "k": k, "url": url, "doc": doc}


    def render_sent(item):
        return div(
            p(f"{query_ui.value}?", klass="font-bold text-2xl"),
            p(item["doc"], klass="text-xl font-semibold text-gray-600"),
            klass="p-4 border rounded-lg shadow-md bg-gray-100",
        )


    aq = list(AnnotationQueue(query_ui.value, n_results=100).make_stream())
    random.shuffle(aq)
    return aq, render_sent


@app.cell
def _(div, p, span):
    def render(doc, sent):
        text = []
        for s in doc["sentences"]:
            text.append(span(s) if s != sent else span(s, style="background-color: yellow;"))
        return div(p(doc["title"], klass="font-bold"), *text)
    return


@app.cell
def _(aq, get_annot, get_current, set_annot, set_current):
    def annotate(outcome):
        set_annot(get_annot() + [{**aq[get_current()], "outcome": outcome}])
        set_current(get_current() + 1)
    return (annotate,)


@app.cell
def _():
    from mohtml import div, span, p, tailwind_css, br

    tailwind_css()
    return br, div, p, span


@app.cell
def _(mo):
    get_annot, set_annot = mo.state([])
    get_current, set_current = mo.state(0)
    return get_annot, get_current, set_annot, set_current


@app.cell
def _(annotate, mo):
    btn_accept = mo.ui.button(label="accept", on_change=annotate, value="accept")
    btn_reject = mo.ui.button(label="reject", on_change=annotate, value="reject")
    btn_skip = mo.ui.button(label="skip", on_change=annotate, value="skip")
    return btn_accept, btn_reject, btn_skip


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""## This is where you annotate""")
    return


@app.cell
def _(mo, set_annot, set_current):
    mo.ui.button(label="reset annot", on_change=lambda d: [set_annot([]), set_current(0)])
    return


@app.cell
def _(mo):
    query_ui = mo.ui.text_area(label="Query", value="data quality").form()
    query_ui
    return (query_ui,)


@app.cell
def _(aq, br, btn_accept, btn_reject, btn_skip, get_current, mo, render_sent):
    mo.vstack(
        [
            render_sent(aq[get_current()]),
            br(),
            mo.hstack([btn_accept, btn_reject, btn_skip]),
        ]
    )
    return


@app.cell
def _(get_annot, mo, pl):
    mo.stop(not len(get_annot()))

    beta_data = (
        pl.DataFrame(get_annot())
        .filter(pl.col("outcome") != "skip")
        .with_columns(score=(pl.col("outcome") == "accept").cast(pl.Int8))
        .group_by("name")
        .agg(pl.len(), pl.sum("score"))
    ).to_dicts()
    return (beta_data,)


@app.cell
def _(beta_data, np):
    import pandas as pd
    import altair as alt
    from scipy.stats import beta

    x = np.linspace(0, 1, 200)
    df = pd.DataFrame(
        {
            "x": np.tile(x, 2),
            "density": np.concatenate(
                [
                    beta.pdf(x, beta_data[0]["score"], beta_data[0]["len"]),
                    beta.pdf(x, beta_data[1]["score"], beta_data[1]["len"]),
                ]
            ),
            "distribution": np.repeat([_["name"] for _ in beta_data], len(x)),
        }
    )

    alt.Chart(df).mark_line().encode(x="x", y="density", color="distribution")
    return


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _():
    return


@app.cell(column=4)
def _(GamepadWidget, mo):
    gamepad = mo.ui.anywidget(GamepadWidget())
    gamepad
    return (gamepad,)


@app.cell
def _(annotate, gamepad, get_i, set_i):
    def observer(change):
        if (get_i() + 150) < change["new"]:
            set_i(change["new"])
            if gamepad.current_button_press == 0:
                annotate("accept")
            elif gamepad.current_button_press == 1:
                annotate("reject")
            elif gamepad.current_button_press == 5:
                annotate("skip")


    gamepad.observe(observer, ["current_timestamp"])
    return


@app.cell
def _(mo):
    get_i, set_i = mo.state(0)
    return get_i, set_i


@app.cell
def _():
    from mopad import GamepadWidget
    return (GamepadWidget,)


@app.cell
def _():
    return


@app.cell(column=5)
def _():
    return


if __name__ == "__main__":
    app.run()
