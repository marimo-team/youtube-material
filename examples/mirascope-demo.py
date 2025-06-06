# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "fastmcp==2.3.4",
#     "marimo",
#     "mcp==1.9.0",
#     "mirascope==1.23.3",
#     "mohtml==0.1.10",
#     "openai==1.78.1",
#     "pydantic==2.11.4",
# ]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""## The setup""")
    return


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import random
    from mirascope import BaseTool, llm
    from pydantic import Field
    import asyncio
    return llm, mo, random


@app.cell
def _(mo):
    get_memory, set_memory = mo.state([])
    return get_memory, set_memory


@app.cell
def _(get_memory, set_memory):
    import time


    def remember(thing: list[str]):
        """This allows you to store a memory of the user."""
        set_memory(get_memory() + thing)
        return "I will remember that!"


    def recall():
        """When the user asks you to remember something, this can retreive the relevant facts."""
        return get_memory()


    def log(msg):
        print(f"{time.strftime('%X')} {msg}")
    return log, recall, remember


@app.cell
def _(llm, log, random, recall, remember):
    def roll_dice(sides: int = 6):
        log(f"Rolling a {sides} sided die.")
        return str(random.randint(1, sides + 1))


    @llm.call(provider="openai", model="gpt-4o-mini", tools=[roll_dice, remember, recall])
    def talk(chatter: str, config: dict = {}) -> str:
        return chatter


    @llm.call(provider="openai", model="gpt-4o-mini")
    def summary(memory: list[str], query: str) -> str:
        prompt = f"You have the following memories: {memory}. The user has this query: {query}. Do not give extra information unless the user is really asking for it. If the user is asking for all memories then give all of them."
        log(f"{prompt=}")
        return prompt


    def chat_func(chatter: str):
        response = talk(chatter)
        if tool := response.tool:
            log(f"We have detected a tool call: {tool.tool_call.function.name}")
            if tool.tool_call.function.name == "recall":
                log(f"{tool.call()=}")
                resp = summary(memory=tool.call(), query=chatter)
                log(f"recall response: {resp.content=}")
                return resp.content
            return tool.call()
        else:
            log("No tool call detected.")
            return response.content
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""## The state + widget""")
    return


@app.cell
def _(mo, print_all):
    chat = mo.ui.chat(print_all)
    chat
    return


@app.cell
def _(mo):
    get_hist, set_hist = mo.state([])
    return get_hist, set_hist


@app.cell
def _(llm, set_hist):
    from mirascope import BaseMessageParam, Messages, prompt_template


    @prompt_template()
    def chat_prompt_mem(history: list[BaseMessageParam]) -> Messages.Type:
        return [
            Messages.System(
                "You have access to memory of a convo. Here's the conversation so far. You are a helpful assistant who can recall relevant memories when asked. You are only allow to remember things, if the user explicitly asks you to remember something. If the user asks you to recall something, you should only recall what is relevant to the current conversation. Do not give extra information unless the user is really asking for it. Completely forget everything else in this conversation.",
            ),
            *history,
        ]


    @llm.call(provider="openai", model="gpt-4o-mini")
    def do_with_memory(history: list[BaseMessageParam]):
        return chat_prompt_mem(history)


    def print_all(msgs):
        msgs = [BaseMessageParam(role=m.role, content=m.content) for m in msgs]
        resp = do_with_memory(msgs)
        set_hist(resp.messages + [resp.message_param])
        return resp
    return (print_all,)


@app.cell
def _():
    return


@app.cell(column=2)
def _():
    from mohtml import div, span, tailwind_css

    tailwind_css()
    return div, span


@app.cell
def _(div, get_hist, span):
    div(
        *[div(
            span(h.content),
            span(h.role, klass="text-gray-500 ml-2")
        ) for h in get_hist()]
    )
    return


@app.cell(column=3)
def _():
    return


if __name__ == "__main__":
    app.run()
