# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anthropic==0.51.0",
#     "chonkie[model2vec]==1.0.7",
#     "chromadb==1.0.9",
#     "diskcache==5.6.3",
#     "marimo",
#     "model2vec==0.5.0",
#     "mohtml==0.1.10",
#     "numpy==2.2.6",
#     "polars==1.29.0",
#     "scikit-learn==1.6.1",
#     "srsly==2.5.1",
# ]
# ///

import marimo

__generated_with = "0.13.10"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""## Setup of cache/db""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import chromadb
    from diskcache import Cache

    client = chromadb.PersistentClient(path="new-chroma-db/")
    cache = {}
    return cache, client


@app.cell
def _():
    from model2vec import StaticModel

    model = StaticModel.from_pretrained("minishlab/potion-base-8M")
    return (model,)


@app.cell
def _(cache, model):
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


    def file_to_items(path):
        urls = []
        for item in srsly.read_jsonl(path):
            embs = model.encode(item["sentences"])
            if item["url"] in cache:
                continue
            cache[item["url"]] = item
            for sent, emb in zip(item["sentences"], embs):
                yield md5_hash(sent), sent, emb, {"url": item["url"]}
    return Path, file_to_items


@app.cell
def _(client):
    from datetime import datetime

    collection = client.get_or_create_collection(
        name="arxiv-collection",
        metadata={
            "description": "Searching in the arxiv-frontpage app",
            "created": str(datetime.now()),
        },
    )
    return (collection,)


@app.cell
def _(cache):
    len(cache)
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""## Adding documents""")
    return


@app.cell
def _(Path, collection, file_to_items, mo):
    from chromadb.errors import DuplicateIDError

    file_generator = Path("../arxiv-frontpage/data/downloads").glob("*.jsonl")

    for _ in mo.status.progress_bar(
        range(200),
        title="Adding data to ChromaDB",
        subtitle="Working!",
        show_eta=True,
        show_rate=True,
    ):
        file = next(file_generator)
        # Process each file's items in batches to ensure consistent lengths
        items = list(file_to_items(file))

        # If there are items to process
        if items:
            # Unzip the items into separate lists
            ids, docs, embs, metadatas = zip(*items)

            # Verify all lists have the same length before upserting
            assert len(ids) == len(docs) == len(embs) == len(metadatas), (
                "Mismatched lengths in data"
            )
            try:
                collection.upsert(
                    documents=list(docs),
                    embeddings=list(embs),
                    metadatas=list(metadatas),
                    ids=list(ids),
                )
            except DuplicateIDError:
                print(f"skipping {file}, has duplicates")
    return (file_generator,)


@app.cell
def _(collection):
    collection.count()
    return


@app.cell
def _(file_generator, file_to_items):
    list(file_to_items(next(file_generator)))
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""## Query""")
    return


@app.cell
def _(mo):
    query_ui = mo.ui.text_area(label="Query", value="data quality").form()
    query_ui
    return (query_ui,)


@app.cell
def _(mo, set_annot, set_current):
    mo.ui.button(label="reset annot", on_change=lambda d: [set_annot([]), set_current(0)])
    return


@app.cell
def _(get_annot):
    import polars as pl

    pl.DataFrame(get_annot(), schema=["sent", "annot"])
    return


@app.cell
def _(collection):
    collection.count()
    return


@app.cell
def _(collection, model, query_ui):
    result = collection.query(model.encode(query_ui.value), n_results=100)
    urls = [_["url"] for _ in result["metadatas"][0]][::-1]
    res_docs = result["documents"][0][::-1]
    return res_docs, urls


@app.cell
def _(div, p, span):
    def render(doc, sent):
        text = []
        for s in doc["sentences"]:
            text.append(span(s) if s != sent else span(s, style="background-color: yellow;"))
        return div(p(doc["title"], klass="font-bold"), *text)
    return (render,)


@app.cell
def _(get_annot, get_current, res_docs, set_annot, set_current):
    def annotate(outcome):
        set_annot(get_annot() + [(res_docs[get_current()], outcome)])
        set_current(get_current() + 1)
    return (annotate,)


@app.cell
def _():
    from mohtml import div, span, p, tailwind_css

    tailwind_css()
    return div, p, span


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


@app.cell
def _():
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""## Annotate""")
    return


@app.cell
def _(
    btn_accept,
    btn_reject,
    btn_skip,
    cache,
    get_current,
    mo,
    render,
    res_docs,
    urls,
):
    mo.vstack(
        [
            render(cache[urls[get_current()]], res_docs[get_current()]),
            mo.hstack([btn_accept, btn_reject, btn_skip]),
        ]
    )
    return


@app.cell
def _(cache, get_max_indices, order, render, res_docs, urls):
    [render(cache[urls[i]], res_docs[i]) for i in get_max_indices(order)]
    return


@app.cell
def _():
    from sklearn.semi_supervised import LabelPropagation

    lp = LabelPropagation()
    return (lp,)


@app.cell
def _(get_annot, lp, model, res_docs, urls):
    order = list(range(len(urls)))
    if len(get_annot()) > 3:
        texts, labels = zip(*get_annot())
        lp.fit(model.encode(texts), labels)
        preds = lp.predict_proba(model.encode(res_docs))
        order = preds[:, lp.classes_ == "accept"]
    return (order,)


@app.cell
def _(order):
    import numpy as np


    def get_max_indices(arr):
        return np.argsort(-np.array(arr).flatten())


    get_max_indices(order)
    return (get_max_indices,)


@app.cell(column=4)
def _():
    return


if __name__ == "__main__":
    app.run()
