import marimo

__generated_with = "0.11.14"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    import polars as pl
    from smartfunc import async_backend
    return async_backend, mo, pl


@app.cell
def _():
    references = ["The Rolling Stones","Fleetwood Mac","Beyoncé","Michael Jackson","Friends","The Sopranos","Breaking Bad","Game of Thrones","The Simpsons","The Office","space", "fast", "slow", "popular", "maths", "complicated", "smart", "supersmart", "The Godfather","Star Wars","Pulp Fiction","The Shawshank Redemption","The Matrix","Titanic","Forrest Gump","The Beatles", "Stranger Things","Seinfeld","Harry Potter","Pride and Prejudice","The Lord of the Rings","Tetris","Super Mario Bros.","The Legend of Zelda","Minecraft","Grand Theft Auto","Pokémon","World of Warcraft","Final Fantasy","Call of Duty", "starcraft", "zerg", "protoss"]
    return (references,)


@app.cell
def _(generate_emoji_a, generate_emoji_b):
    async def make_pair(reference):
        ex1 = (await generate_emoji_a(reference))["emoji"]
        ex2 = (await generate_emoji_b(reference))["emoji"]
        return {"reference": reference, "emoji_a": ex1, "emoji_b": ex2}
    return (make_pair,)


@app.cell
def _():
    from mohtml import p, div, tailwind_css, br, span

    tailwind_css()
    return br, div, p, span, tailwind_css


@app.cell
def _(get_example):
    get_example()
    return


@app.cell
def _(br, buttonstack, div, get_example, p, span):
    ex = get_example()

    div(
        p(
            f"'{ex['reference']}'", 
            klass="text-2xl font-bold p-2 text-center"
         ),
        br(),
        p(
            span(f"{ex['emoji_a']}", klass="p-4 bg-lime-200 rounded-lg"),
            span("vs.", klass="px-4"),
            span(f"{ex['emoji_b']}", klass="p-4 bg-lime-200 rounded-lg"),
            klass="text-2xl font-bold p-2 text-center"
         ),
        br(),
        buttonstack,
        klass="p-8 bg-lime-100 rounded-lg",
    )
    return (ex,)


@app.cell
def _(gen, get_example, get_labels, set_example, set_labels):
    def undo():
        set_labels(get_labels()[:-2])

    def add_label(lab):
        new_state = get_labels() + [{"example": get_example(), "annotation": lab}]
        set_labels(new_state)
        set_example(next(gen))
    return add_label, undo


@app.cell
def _(gen, mo):
    get_labels, set_labels = mo.state([])
    get_example, set_example = mo.state(next(gen))
    return get_example, get_labels, set_example, set_labels


@app.cell
def _(add_label, mo, undo):
    btn_yes  = mo.ui.button(value=0, label=f"1st - j", keyboard_shortcut=f"Ctrl-j", on_click=lambda d: d + 1, on_change=lambda d: add_label("a")) 
    btn_no   = mo.ui.button(value=0, label=f"2nd - k", keyboard_shortcut=f"Ctrl-k", on_click=lambda d: d + 1, on_change=lambda d: add_label("b")) 
    btn_skip = mo.ui.button(value=0, label=f"skip - l", keyboard_shortcut=f"Ctrl-l", on_click=lambda d: d + 1, on_change=lambda d: add_label("skip")) 
    btn_undo = mo.ui.button(value=0, label=f"undo - ;", keyboard_shortcut=f"Ctrl-;", on_click=lambda d: d + 1, on_change=lambda d: undo()) 

    buttonstack = mo.hstack([btn_yes, btn_no, btn_skip, btn_undo])
    return btn_no, btn_skip, btn_undo, btn_yes, buttonstack


@app.cell
def _(get_labels, pl):
    pl.DataFrame(get_labels()).reverse()
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _(async_backend):
    from pydantic import BaseModel, ConfigDict
    from dotenv import load_dotenv

    load_dotenv(".env")


    class WithSummary(BaseModel):
        summary: str
        emoji: str
        model_config = ConfigDict(extra='ignore')


    class WithoutSummary(BaseModel):
        emoji: str
        model_config = ConfigDict(extra='ignore')


    llmify = async_backend("gpt-4o-mini", temperature=0.5)

    @llmify
    def generate_emoji_a(phrase: str) -> WithSummary:
        """Generate a sequence of emoji that describes the following phrase: '{{ phrase }}'. Also add a small summary that helps describe why this sequence of emoji is fitting for the phrase. Use as many emoji as you need."""
        pass


    @llmify
    def generate_emoji_b(phrase: str) -> WithoutSummary:
        """Generate a sequence of emoji that describes the following phrase: '{{ phrase }}'. Use as many emoji as you need."""
        pass
    return (
        BaseModel,
        ConfigDict,
        WithSummary,
        WithoutSummary,
        generate_emoji_a,
        generate_emoji_b,
        llmify,
        load_dotenv,
    )


@app.cell
async def _(make_pair, references):
    import asyncio
    from mosync import async_map_with_retry


    async def delayed_double(x):
        await asyncio.sleep(1)
        return x * 2

    results = await async_map_with_retry(
        references,
        make_pair, 
        max_concurrency=10, 
        description="Generating emoji data"
    )
    return async_map_with_retry, asyncio, delayed_double, results


@app.cell
def _(get_labels, mo, pl):
    mo.stop(len(get_labels()) == 0)
    pl.DataFrame(get_labels()).group_by("annotation").len()
    return


@app.cell
def _():
    return


@app.cell(column=2)
def _(results):
    gen = (r.result for r in results)
    return (gen,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
