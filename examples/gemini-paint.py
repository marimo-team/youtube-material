# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "google-genai==1.10.0",
#     "marimo",
#     "mohtml==0.1.5",
#     "mopaint==0.2.1",
#     "pillow==11.1.0",
#     "protobuf==6.30.2",
#     "python-dotenv==1.1.0",
# ]
# ///

import marimo

__generated_with = "0.14.0"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _():
    from mohtml import p, tailwind_css, div, span
    return


@app.cell
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
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]

        # Decode base64 string
        img_data = base64.b64decode(base64_str)

        # Create PIL Image object
        img = Image.open(BytesIO(img_data))

        return img

    def pil_to_image_bytes(pil_image):
        """Convert a PIL Image object to bytes.

        Args:
            pil_image (PIL.Image): The PIL Image object

        Returns:
            bytes: The image converted to bytes
        """
        # Create a BytesIO object
        img_byte_arr = BytesIO()

        # Convert RGBA to RGB if the image has an alpha channel
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')

        # Save the image to the BytesIO object in JPEG format
        pil_image.save(img_byte_arr, format='JPEG')

        # Get the bytes from the BytesIO object
        image_bytes = img_byte_arr.getvalue()

        return image_bytes
    return base64_to_pil, pil_to_image_bytes


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


@app.cell(column=1)
def _(Paint, mo):
    paint = mo.ui.anywidget(Paint())
    paint
    return (paint,)


@app.cell
def _(mo):
    text_input = mo.ui.text_area(label="prompt", value="")
    text_input
    return (text_input,)


@app.cell(column=2)
def _(base64_to_pil, paint):
    base64_to_pil(paint.value["base64"])
    return


@app.cell(hide_code=True)
def _(client, mo, paint, pil_to_image_bytes, run_btn, text_input, types):
    mo.stop(not run_btn.value, "Hit run button first")

    # from google.genai import types


    response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
      types.Part.from_bytes(
        data=pil_to_image_bytes(paint.get_pil()),
        mime_type='image/jpeg',
      ),
      text_input.value
    ]
    )

    print(response.text)

    # _response = client.models.generate_content(
    #     model="gemini-2.0-flash",
    #     contents=[text_input.value, base64_to_pil(paint.value["base64"])],
    #     config=types.GenerateContentConfig(response_modalities=["Text"]),
    # )

    # for part in _response.candidates[0].content.parts:
    #     if part.text is not None:
    #         print(part.text)
    #     image = None
    #     if part.inline_data is not None:
    #         image = Image.open(BytesIO((part.inline_data.data)))

    # div(span("Prompt response:", klass="font-bold"), p(part.text))
    return


@app.cell
def _(mo):
    run_btn = mo.ui.run_button()
    run_btn
    return (run_btn,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(column=3)
def _():
    return


if __name__ == "__main__":
    app.run()
