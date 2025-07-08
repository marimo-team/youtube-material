# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "drawdata==0.3.8",
#     "logfire==3.22.1",
#     "marimo",
#     "pydantic==2.11.4",
#     "pydantic-ai==0.2.6",
#     "pydantic-graph==0.2.6",
# ]
# ///

import marimo

__generated_with = "0.14.10"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from dataclasses import dataclass

    from pydantic import BaseModel, Field
    from pydantic_ai import Agent, RunContext, ModelRetry
    from enum import Enum
    from pydantic_graph import End
    from pprint import pprint
    from typing import Optional, Literal
    return Agent, BaseModel, Field, Literal


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Introduction 

    This bit is here to allow one to explain the basics of the syntax/objects.
    """
    )
    return


@app.cell
def _():
    import logfire

    logfire.configure()
    logfire.instrument_pydantic_ai()
    return


@app.cell
def _(BaseModel, Field, Literal):
    class Pizza(BaseModel):
        flavour: str
        """The type of pizza that the user wants to order"""
        size: Literal["small", "medium", "large"] = Field(
            description="The size of pizza that the user wants to order"
        )
    return (Pizza,)


@app.cell
def _(Agent, Pizza):
    agent = Agent(
        "openai:gpt-4.1",
        output_type=list[Pizza] | str,
        instructions="You are here to detect pizza purchases from the user. If the user does not ask for a pizza then you have to ask.",
    )


    @agent.tool_plain
    def flavour_by_location(location: str):
        """Always check the location of the user to make sure they order a correct pizza."""
        if len(location) > 5:
            return ["veggie"]
        else:
            return ["chocolatte", "nutella"]
    return (agent,)


@app.cell
def _(agent):
    from pydantic_ai.messages import ModelResponse, ModelRequest, UserPromptPart, TextPart


    async def handle_pydantic_ai(messages):
        message_history = []
        for m in messages:
            if m.role == "assistant":
                message_history.append(ModelResponse(parts=[TextPart(m.content)]))
            if m.role == "user":
                message_history.append(ModelRequest(parts=[UserPromptPart(m.content)]))

        resp = await agent.run(message_history=message_history)
        print(resp)
        return resp.output
    return (handle_pydantic_ai,)


@app.cell(column=1)
def _(handle_pydantic_ai, mo):
    chat = mo.ui.chat(handle_pydantic_ai)
    chat
    return (chat,)


@app.cell(column=2)
def _():
    return


@app.cell
def _(chat):
    chat.value
    return


if __name__ == "__main__":
    app.run()
