# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==5.5.0",
#     "jupyter-scatter==0.21.1",
#     "marimo",
#     "mohtml==0.1.2",
#     "numpy==2.2.3",
#     "openai==1.65.4",
#     "polars==1.24.0",
#     "requests==2.32.3",
#     "vegafusion==2.0.2",
#     "vl-convert-python",
# ]
# ///

import marimo

__generated_with = "0.11.14"
app = marimo.App(width="medium")


app._unparsable_cell(
    r"""
        from mohtml import div, p, tailwind_css, br
    import marimo as mo
    import requests
    from pathlib import Path
    import numpy as np
    import polars as pl
    import altair as alt
    """,
    name="_"
)


@app.cell
def _(tailwind_css):
    tailwind_css()
    return


@app.cell
def _(Path, mo, pl, requests):
    emb_path = Path("mtg_embeddings.parquet")
    with mo.status.spinner(subtitle="Fetching embedding data") as _s:
        if not emb_path.exists():
            url = "https://huggingface.co/datasets/minimaxir/mtg-embeddings/resolve/main/mtg_embeddings.parquet"

            with requests.get(url, stream=True) as response:
                response.raise_for_status()  # Check for request errors
                with open(emb_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
        df_emb = pl.read_parquet(Path("mtg_embeddings.parquet"))
    return chunk, df_emb, emb_path, f, response, url


@app.cell
def _(mo):
    text_input = mo.ui.text(label="Search for a card")
    text_input
    return (text_input,)


@app.cell
def _(df_emb, mo, pl, text_input):
    tbl = mo.ui.table(df_emb.filter(pl.col("name").str.to_lowercase().str.contains(text_input.value)))
    tbl
    return (tbl,)


@app.cell
def _(requests):
    SCRYFALL_URI = "https://api.scryfall.com"

    def get_img_url(query_card_id):
        headers = {"User-Agent": "Related Card Image/1.0", "Accept": "*/*"}
        card_uri = f"{SCRYFALL_URI}/cards/{query_card_id}"

        r = requests.get(card_uri, headers=headers)

        return r.json()["image_uris"]["png"]
    return SCRYFALL_URI, get_img_url


@app.cell
def _(br, div, get_img_url, mo, p, tbl):
    div(
        p("Query cards", klass="font-bold text-xl"),
        br(),
        mo.hstack([
            mo.image(get_img_url(sid), height=500) for sid in tbl.value["scryfallId"]
        ])
    )
    return


@app.cell
def _(tbl):
    query_ids = set(tbl.value["scryfallId"].to_list())
    return (query_ids,)


@app.cell
def _(df_emb, np, tbl):
    def fast_dot_product(query, matrix, k=10):
        dot_products = query @ matrix.T

        idx = np.argpartition(dot_products, -k)[-k:]
        idx = idx[np.argsort(dot_products[idx])[::-1]]

        score = dot_products[idx]

        return idx, score

    idx, score = fast_dot_product(
        query=np.sum(tbl.value["embedding"].to_numpy(), axis=0), 
        matrix=df_emb["embedding"].to_numpy()
    )
    return fast_dot_product, idx, score


@app.cell
def _(br, df_emb, div, get_img_url, idx, mo, p, query_ids):
    results = [mo.image(get_img_url(sid), height=500) for sid in df_emb[idx]["scryfallId"] if sid not in query_ids]

    def list_to_grid(long_list, max_width=3):
        return mo.hstack(
                [mo.vstack(long_list[i:i + max_width]) for i in range(0, len(long_list), max_width)]
        )

    div(
        p("Retreived cards", klass="font-bold text-xl"),
        br(),
        list_to_grid(results)
    )
    return list_to_grid, results


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
