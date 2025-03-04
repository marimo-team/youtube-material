import marimo

__generated_with = "0.11.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo 
    import typer
    from typing import List
    from pathlib import Path

    app = typer.Typer(name="demo", add_completion=False, help="This is a demo app.")


    @app.command()
    def hello_world(name):
        """Say hello"""
        print(f"hello {name}!")

    @app.command()
    def goodbye_world(name):
        """Say goodbye"""
        print(f"goodbye {name}!")

    @app.command()
    def add(n1: int = typer.Argument(..., help="An integer"),
            n2: int = typer.Argument(1, help="An integer")):
        """Add two numbers"""
        print(n1 + n2)

    def check_file_exists(paths):
        for p in paths:
            if not p.exists():
                print(f"The path you've supplied {p} does not exist.")
                raise typer.Exit(code=1)
        return paths

    @app.command()
    def word_count(paths: List[Path] = typer.Argument(...,
                                                    help="The file to count the words in.",
                                                    callback=check_file_exists)):
        """Counts the number of words in a file"""
        for p in paths:
            texts = p.read_text().split("\n")
            n_words = len(set(w for t in texts for w in t.split(" ")))
            print(f"In total there are {n_words} words in {p}.")

    @app.command()
    def talk(text: str = typer.Argument(..., help="The text to type."),
            repeat: int = typer.Option(1, help="Number of times to repeat."),
            loud: bool = typer.Option(False, is_flag=True)):
        """Talks some text below"""
        if loud:
            text = text.upper()
        for _ in range(repeat):
            print(text)
    return (
        List,
        Path,
        add,
        app,
        check_file_exists,
        goodbye_world,
        hello_world,
        mo,
        talk,
        typer,
        word_count,
    )


@app.cell
def _(app, mo):
    if mo.app_meta().mode == "script":
        app()
    return


@app.cell
def _():
    from wigglystuff import Matrix
    return (Matrix,)


@app.cell
def _(Matrix):
    mat = Matrix(rows=2)
    return (mat,)


@app.cell
def _(mat, mo):
    widget = mo.ui.anywidget(mat)
    return (widget,)


@app.cell
def _(widget):
    widget
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
