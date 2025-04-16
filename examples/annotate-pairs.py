# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "diskcache==5.6.3",
#     "llm==0.24.2",
#     "marimo",
#     "mohtml==0.1.5",
#     "moutils==0.1.1",
#     "polars==1.27.1",
#     "python-dotenv==1.1.0",
#     "smartfunc==0.2.0",
# ]
# ///

import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from diskcache import Cache
    from smartfunc import backend, get_backend_models
    from dotenv import load_dotenv
    import llm

    load_dotenv(".env")
    return Cache, backend, get_backend_models, llm, load_dotenv, mo


@app.cell
def _(Cache, llm, prompts, topics):
    model = llm.get_model("gpt-4")
    cache = Cache("naming-prompts")

    @cache.memoize()
    def haiku(prompt, topic, seed):
        return model.prompt(prompt.format(topic=topic, seed=seed)).text()

    for _t in topics:
        for _p in prompts:
            for _i in range(4):
                haiku(prompt=_p, topic=_t, seed=_i)
    return cache, haiku, model


@app.cell
def _():
    prompts = [
        "seed={seed} Write me a haiku about {topic}",
        "seed={seed} Write me a funny haiku about {topic} that rhymes."
    ]
    topics = ["python library", "new databases"]
    return prompts, topics


@app.cell
def _(cache):
    from collections import defaultdict

    cache_out = [(k[3], k[5], k[7], cache[k]) for k in cache.iterkeys()]
    stream = []

    for prompt, seed, topic, result in cache_out:
        stream.append({
            "prompt": prompt, 
            "inputs": {"topic": topic},
            "result": result
        })

    stream
    return cache_out, defaultdict, prompt, result, seed, stream, topic


@app.cell
def _(stream):
    import polars as pl

    df_stream = pl.DataFrame(stream).group_by("prompt", "inputs").agg(pl.col("result").explode())
    annot_stream = (_ for _ in 
        df_stream
          .join(df_stream, on=["inputs"], how="left")
          .select(
              "inputs",
              pl.col("prompt").alias("prompt_left"),
              pl.col("result").alias("result_left"),
              "prompt_right",
              "result_right"
          )
          .filter(pl.col("prompt_left") != pl.col("prompt_right"))
          .explode("result_left", "result_right")
          .sample(fraction=1, shuffle=True)
          .to_dicts()
    )
    return annot_stream, df_stream, pl


@app.cell
def _():
    # pl.DataFrame(annot_stream)
    return


@app.cell
def _(btn_left, btn_right, btn_skip, get_example, mo):
    from mohtml import div

    mo.vstack([
        mo.md("## Which joke is better?"),
        mo.hstack([
            get_example()["result_left"], 
            get_example()["result_right"]
        ]),
        mo.hstack([
            btn_left,
            btn_skip,
            btn_right
        ])
    ])
    return (div,)


@app.cell
def _():
    return


@app.cell
def _(mo, update):
    btn_left = mo.ui.button(label="left", keyboard_shortcut="Ctrl-j", on_change=lambda d: update("left"))
    btn_skip = mo.ui.button(label="skip", keyboard_shortcut="Ctrl-k", on_change=lambda d: update("skip"))
    btn_right = mo.ui.button(label="right", keyboard_shortcut="Ctrl-l", on_change=lambda d: update("right"))
    return btn_left, btn_right, btn_skip


@app.cell
def _(annot_stream, mo):
    get_example, set_example = mo.state(next(annot_stream))
    get_annot, set_annot = mo.state([])

    def update(outcome):
        ex = get_example()
        ex["outcome"] = outcome
        set_annot(get_annot() + [ex])
        set_example(next(annot_stream))
    return get_annot, get_example, set_annot, set_example, update


@app.cell
def _(get_annot):
    get_annot()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
