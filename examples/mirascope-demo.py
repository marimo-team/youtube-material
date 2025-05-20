# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "fastmcp==2.3.4",
#     "marimo",
#     "mcp==1.9.0",
#     "mirascope==1.23.3",
#     "openai==1.78.1",
#     "pydantic==2.11.4",
# ]
# ///

import marimo

__generated_with = "0.13.10"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    from pathlib import Path
    import marimo as mo
    import random
    from mirascope import BaseTool, llm, prompt_template
    from pydantic import Field
    import asyncio
    return llm, mo, random


@app.cell
def _(mo):
    get_memory, set_memory = mo.state([])
    return get_memory, set_memory


@app.cell
def _(get_memory):
    get_memory()
    return


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
        print(f"{time.strftime("%X")} {msg}")
    return log, recall, remember


@app.cell
def _(llm, log, random, recall, remember):
    def roll_dice(sides:int = 6):
        return str(random.randint(1, sides + 1))

    @llm.call(provider="openai", model="gpt-4o-mini", tools=[roll_dice, remember, recall])
    def talk(chatter: str, config: dict={}) -> str: 
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
                print(resp)
                return resp.content
            return tool.call()
        else:
            return response.content
    return (chat_func,)


@app.cell
def _(chat_func):
    chat_func
    return


@app.cell(column=1)
def _(chat_func, mo):
    chat = mo.ui.chat(lambda messages: chat_func(messages[-1].content))
    chat
    return


if __name__ == "__main__":
    app.run()
