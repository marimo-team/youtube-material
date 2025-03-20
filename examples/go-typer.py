# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anthropic==0.49.0",
#     "click==8.1.8",
#     "marimo",
#     "rich==13.9.4",
#     "typer==0.15.2",
# ]
# ///

import marimo

__generated_with = "0.11.22"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import typer
    from typing import Optional

    app = typer.Typer(help="A simple CLI application")

    @app.command()
    def hello(name: str, count: int = 1, formal: bool = False):
        """
        Say hello to NAME COUNT times, with an option to be formal.
        """
        greeting = "Hello" if not formal else "Greetings"
        for _ in range(count):
            print(f"{greeting}, {name}!")

    @app.command()
    def goodbye(name: str, formal: bool = False):
        """
        Say goodbye to NAME, with an option to be formal.
        """
        farewell = "Bye" if not formal else "Farewell"
        print(f"{farewell}, {name}!")
    return Optional, app, goodbye, hello, typer


@app.cell
def _(app, mo):
    if mo.app_meta().mode == "script":  
        app()
    return


@app.cell
def _():
    import click

    @click.group(help="A simple CLI application")
    def cli():
        """A simple CLI application."""
        pass

    @cli.command()
    @click.argument("name")
    @click.option("--count", default=1, help="Number of times to say hello.")
    @click.option("--formal/--no-formal", default=False, help="Use formal greeting.")
    def click_hello(name, count, formal):
        """Say hello to NAME COUNT times, with an option to be formal."""
        greeting = "Hello" if not formal else "Greetings"
        for _ in range(count):
            click.echo(f"{greeting}, {name}!")

    @cli.command()
    @click.argument("name")
    @click.option("--formal/--no-formal", default=False, help="Use formal farewell.")
    def click_goodbye(name, formal):
        """Say goodbye to NAME, with an option to be formal."""
        farewell = "Bye" if not formal else "Farewell"
        click.echo(f"{farewell}, {name}!")
    return cli, click, click_goodbye, click_hello


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
