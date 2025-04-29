# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "google-genai==1.10.0",
#     "marimo",
#     "mohtml==0.1.5",
#     "mopaint==0.1.5",
#     "pillow==11.1.0",
#     "protobuf==6.30.2",
#     "python-dotenv==1.1.0",
# ]
# ///

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="columns", layout_file="layouts/gemini-paint.grid.json")


@app.cell(column=0)
def _():
    import marimo as mo
    from mopaint import Paint
    return Paint, mo


@app.cell
def _():
    import base64
    from io import BytesIO
    from PIL import Image

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
    return BytesIO, Image, base64_to_pil


@app.cell
def _():
    from google import genai
    from google.genai import types
    from dotenv import load_dotenv
    import PIL.Image
    import os

    load_dotenv(".env")

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    return client, types


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    from mohtml import p, tailwind_css, div

    checkbox = mo.ui.checkbox(label="background setting")
    checkbox
    return checkbox, p


@app.cell
def _(Paint, checkbox, mo):
    paint = mo.ui.anywidget(Paint(keep_background=checkbox.value))
    paint
    return (paint,)


@app.cell
def _(mo):
    text_input = mo.ui.text_area(label="prompt", value="Turns this doodle and make it realisitic looking")
    text_input
    return (text_input,)


@app.cell(column=2)
def _(base64_to_pil, paint):
    base64_to_pil(paint.value["base64"])
    return


@app.cell
def _(base64_to_pil, client, mo, p, paint, run_btn):
    mo.stop(not run_btn.value, "Hit run button first")

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=["What is this image?", base64_to_pil(paint.value["base64"])])

    p(response.text)
    return


@app.cell
def _(mo):
    run_btn = mo.ui.run_button()
    run_btn
    return (run_btn,)


@app.cell
def _(
    BytesIO,
    Image,
    base64_to_pil,
    client,
    mo,
    p,
    paint,
    run_btn,
    text_input,
    types,
):
    mo.stop(not run_btn.value, "Hit run button first")

    _response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[text_input.value, base64_to_pil(paint.value["base64"])],
        config=types.GenerateContentConfig(
          response_modalities=['Text']
        )
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
    return


if __name__ == "__main__":
    app.run()
