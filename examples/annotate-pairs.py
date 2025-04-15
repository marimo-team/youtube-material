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
def _(Cache, llm):
    model = llm.get_model("gpt-4")
    cache = Cache("joke-prompts")

    @cache.memoize()
    def joke(prompt, topic, seed):
        return model.prompt(prompt.format(topic=topic, seed=seed)).text()
    return cache, joke, model


@app.cell
def _():
    prompts = [
        "seed={seed} Write me a joke about {topic}",
        "seed={seed} Write me a hilarous joke about {topic} that would really do well on social media. Go against hype and be a bit self deprecating."
    ]
    topics = ["python", "databases"]
    return prompts, topics


@app.cell
def _(joke, prompts, topics):
    for _t in topics:
        for _p in prompts:
            for _i in range(4):
                joke(prompt=_p, topic=_t, seed=_i)
    return


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
def _(annot_stream, pl):
    pl.DataFrame(annot_stream)
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
def _(get_annot, pl):
    pl.DataFrame(get_annot())
    return


@app.cell
def _(mo):
    mo.iframe("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Key Press Detector</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f0f0f0;
            }
            .container {
                text-align: center;
                padding: 2rem;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                width: 80%;
                max-width: 500px;
            }
            .key-display {
                font-size: 5rem;
                font-weight: bold;
                height: 120px;
                display: flex;
                justify-content: center;
                align-items: center;
                margin: 1rem 0;
                background-color: #f8f8f8;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
            .info {
                margin-bottom: 1rem;
                color: #555;
            }
            .info-meta {
                font-size: 0.8rem;
                color: #888;
                margin-top: 2rem;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Key Press Detector</h1>
            <div class="info">Press any key on your keyboard</div>
            <div class="key-display" id="keyDisplay">?</div>
            <div id="keyInfo">Press a key to start</div>
            <div class="info-meta">Click anywhere on the page first if keys aren't detected</div>
        </div>

        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const keyDisplay = document.getElementById('keyDisplay');
                const keyInfo = document.getElementById('keyInfo');

                document.addEventListener('keydown', function(event) {
                    // Prevent default behavior for some keys
                    event.preventDefault();

                    // Display the key
                    if (event.key === ' ') {
                        keyDisplay.textContent = 'Space';
                    } else if (event.key === 'Escape') {
                        keyDisplay.textContent = 'Esc';
                    } else if (event.key.length === 1) {
                        keyDisplay.textContent = event.key;
                    } else {
                        keyDisplay.textContent = event.key;
                    }

                    // Show key information
                    keyInfo.textContent = `Key: ${event.key} | Code: ${event.code} | KeyCode: ${event.keyCode}`;
                });
            });
        </script>
    </body>
    </html>
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
