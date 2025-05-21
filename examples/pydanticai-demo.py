# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pydantic==2.11.4",
#     "pydantic-ai==0.2.6",
#     "pydantic-graph==0.2.6",
# ]
# ///

import marimo

__generated_with = "0.13.10"
app = marimo.App(width="full")


@app.cell
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
    return (
        Agent,
        BaseModel,
        End,
        Field,
        Literal,
        Optional,
        RunContext,
        dataclass,
        pprint,
    )


@app.cell
async def _(Agent):
    _agent = Agent("openai:gpt-4o")

    nodes = []
    async with _agent.iter("What is the capital of France?") as agent_run:
        async for node in agent_run:
            nodes.append(node)
    nodes
    return (agent_run,)


@app.cell
def _(agent_run):
    print(agent_run.result.output)
    return


@app.cell
async def _(Agent, End, Pizza, pprint):
    _agent = Agent("openai:gpt-4o", output_type=Pizza)


    async def _main():
        async with _agent.iter("I want to order one medium hawai pizzas.") as agent_run:
            node = agent_run.next_node

            all_nodes = [node]

            # Drive the iteration manually:
            while not isinstance(node, End):
                node = await agent_run.next(node)
                # I could try and intervene here. I should check the store the user wants to order from
                # and I should also check if the pizza is available that that store. Python can do it
                # but PydanticAI offers another mechanism for this.
                pprint(node)
                all_nodes.append(node)

            return all_nodes


    resp = await _main()
    return


@app.cell
def _(BaseModel, Field, Literal, Optional):
    class Pizza(BaseModel):
        kind: Optional[str] = Field(
            description="The type of pizza that the user wants to order"
        )
        size: Optional[Literal["small", "medium", "large"]] = Field(
            description="The size of pizza that the user wants to order"
        )


    class PizzaOrder(BaseModel):
        pizzas: Optional[list[Pizza]]
    return Pizza, PizzaOrder


@app.cell
def _(Agent, Literal, Optional, PizzaOrder, RunContext, dataclass):
    from pydantic_ai.messages import ModelMessage

    @dataclass
    class OrderDependencies:
        customer_id: int
        allowed_pizzas: Literal["hawai", "veggie", "pepperoni"]
        order: Optional[PizzaOrder]

    pizza_agent = Agent(
        "openai:gpt-4o",
        output_type=PizzaOrder,
        deps_type=PizzaOrder, 
        system_prompt="Your need to figure out what the user is trying to order. It could be that a user is changing their order with new information",
    )

    order_agent = Agent(
        "openai:gpt-4o",
        output_type=PizzaOrder | str,
        deps_type=PizzaOrder,
        system_prompt="It is your job to figure out if the order is complete or if the user needs to provide additional information. If the order is complete you can just pass the pizza order that we received.",
    )

    @order_agent.system_prompt
    def what_to_check_for(ctx: RunContext[OrderDependencies]) -> str:
        possible_pizzas: list[str] = ctx.deps.allowed_pizzas
        return f"These are the possible pizzas for the store: {possible_pizzas}. It could be that we have to do some fuzzy matching. Only do the fuzzy matching if it is clear that we should match, if there is no match we can also leave the pizza kind open. If the user gives us new information, update our belief but do not throw away old preferences. If we know the size, but get a new kind, we need to remember the old size."

    @order_agent.system_prompt
    def what_to_check_for(ctx: RunContext[OrderDependencies]) -> str:
        possible_pizzas: list[str] = ctx.deps.allowed_pizzas
        return f"These are the possible pizzas for the store: {possible_pizzas}. This is the order that we received: {ctx.deps.order.model_dump()}. You need to check if this order is possible in our store. If not, very briefly explain why and try to be helpful."
    return OrderDependencies, order_agent, pizza_agent


@app.cell
def _(OrderDependencies, mo, order_agent, pizza_agent):
    deps = OrderDependencies(customer_id=123, allowed_pizzas=["hawai", "veggie"], order=None)
    get_state, set_state = mo.state(None)

    async def do_a_turn(msg):
        print(msg)
        deps.order = get_state()
        r1 = await pizza_agent.run(msg, deps=deps)
        deps.order = r1.output
        set_state(r1.output)
        r2 = await order_agent.run(
            f"Is there any issue with my current order? {deps.order.model_dump()}.", deps=deps
        )
        msg_out = r2.output
        if not isinstance(r2.output, str):
            msg_out = f"All good. Will proceed with the order for {r2.output.model_dump()}"
            set_state(r2.output)
        return msg_out
    return do_a_turn, get_state


@app.cell
def _(get_state):
    get_state()
    return


@app.cell
def _(do_a_turn, mo):
    chat = mo.ui.chat(lambda messages: do_a_turn(messages[-1].content))
    chat
    return (chat,)


@app.cell
def _(chat):
    chat.value
    return


if __name__ == "__main__":
    app.run()
