# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "flowshow==0.2.2",
#     "marimo",
#     "mofresh==0.2.2",
#     "pydantic==2.11.4",
# ]
# ///

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md("""Flowshow provides a `@task` decorator that helps you track and visualize the execution of your Python functions. Here's how to use it:""")
    return


@app.cell
def _():
    import time
    import random
    import uuid
    from pydantic import BaseModel
    from typing import List
    from flowshow import task, add_artifacts, info, debug, warning, error, span
    return error, info, task, time, uuid, warning


@app.cell
def _(mo):
    get_state, set_state = mo.state({
        "current_id": None, 
        "children": []
    })
    return


@app.cell
def _(uuid):
    def fancy(func): 
        my_id = str(uuid.uuid4())
        my_id
    return


@app.cell
async def _(error, info, task, time, warning):
    import asyncio

    @task
    async def async_sleep(seconds: float, name: str) -> str:
        """Asynchronous sleep function that returns a message after completion"""
        info("it works, right?")
        await asyncio.sleep(seconds)
        info("it did!")
        return f"{name} finished sleeping for {seconds} seconds"

    @task
    async def run_concurrent_tasks():
        """Run multiple sleep tasks concurrently"""
        start_time = time.time()

        # Create multiple sleep tasks
        tasks = [
            async_sleep(2, "Task 1"),
            async_sleep(1, "Task 2"),
            async_sleep(3, "Task 3")
        ]

        # Run tasks concurrently and gather results
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        # Return results and timing information
        return {
            "results": results,
            "total_time": f"Total execution time: {total_time:.2f} seconds"
        }

    @task 
    async def run_many_nested():
        info("About to start task 1")
        await run_concurrent_tasks()
        info("About to start task 2")
        await run_concurrent_tasks()
        warning("They both ran!")
        error("They both ran!")

    await run_many_nested()
    return (run_many_nested,)


@app.cell
def _(mo, run_many_nested):
    mo.iframe(run_many_nested.last_run.render())
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
