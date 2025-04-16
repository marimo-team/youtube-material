# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "google-genai==1.10.0",
#     "marimo",
#     "mohtml==0.1.5",
#     "modraw==0.1.14",
#     "pillow==11.1.0",
#     "protobuf==6.30.2",
#     "python-dotenv==1.1.0",
#     "pydantic==2.11.3",
# ]
# ///

import marimo

__generated_with = "0.12.8"
app = marimo.App(width="columns", layout_file="layouts/gemini-draw.grid.json")


@app.cell(column=0)
def _():
    import marimo as mo
    from modraw import Draw
    return Draw, mo


@app.cell
def _():
    import base64
    from io import BytesIO
    from PIL import Image
    from pydantic import BaseModel


    def base64_to_pil(base64_str):
        """Convert a base64 string to PIL Image object.

        Args:
            base64_str (str): The base64 string of the image

        Returns:
            PIL.Image: The converted PIL Image object
        """
        # Remove the data URL prefix if it exists
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[1]

        # Decode base64 string
        img_data = base64.b64decode(base64_str)

        # Create PIL Image object
        img = Image.open(BytesIO(img_data))

        return img
    return BaseModel, BytesIO, Image, base64, base64_to_pil


@app.cell
def _():
    from google import genai
    from google.genai import types
    from dotenv import load_dotenv
    import PIL.Image
    import os

    load_dotenv(".env")

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    return PIL, client, genai, load_dotenv, os, types


@app.cell
def _(base64_to_pil, client, mo, p, paint, run_btn):
    mo.stop(not run_btn.value, "Hit run button first")

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=["What is this image?", base64_to_pil(paint.value["base64"])])

    p(response.text)
    return (response,)


@app.cell
def _():
    from mohtml import p
    return (p,)


@app.cell
def _(base64_to_pil, paint):
    base64_to_pil(paint.value["base64"])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Something we could consider adding here: 

        ```python
        from google import genai
        from pydantic import BaseModel


        class Recipe(BaseModel):
          recipe_name: str
          ingredients: list[str]


        client = genai.Client(api_key="GEMINI_API_KEY")
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents='List a few popular cookie recipes. Be sure to include the amounts of ingredients.',
            config={
                'response_mime_type': 'application/json',
                'response_schema': list[Recipe],
            },
        )
        # Use the response as a JSON string.
        print(response.text)

        # Use instantiated objects.
        my_recipes: list[Recipe] = response.parsed
        ```
        """
    )
    return


@app.cell(column=1)
def _(Draw, mo):
    paint = mo.ui.anywidget(Draw())
    paint
    return (paint,)


@app.cell
def _(mo):
    run_btn = mo.ui.run_button()
    run_btn
    return (run_btn,)


@app.cell
def _(mo):
    text_input = mo.ui.text_area(label="prompt", value="parse out the json structure out of the drawn dag here")
    text_input
    return (text_input,)


@app.cell
def _(
    BaseModel,
    BytesIO,
    Image,
    base64_to_pil,
    client,
    mo,
    p,
    paint,
    run_btn,
    text_input,
):
    mo.stop(not run_btn.value, "Hit run button first")

    class Graph(BaseModel):
      nodes: list[str]
      links: list[dict]


    _response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=[text_input.value, base64_to_pil(paint.value["base64"])],
        config={
            'response_mime_type': 'application/json',
        }
    )


    for part in _response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        image = None
        if part.inline_data is not None:
            image = Image.open(BytesIO((part.inline_data.data)))

    mo.vstack([
        p(part.text),
        image
    ])
    return Graph, image, part


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
