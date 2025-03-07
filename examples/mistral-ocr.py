# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "mistralai==1.5.1",
#     "mohtml==0.1.2",
#     "parse==1.20.2",
#     "python-dotenv==1.0.1",
# ]
# ///

import marimo

__generated_with = "0.11.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from mistralai import Mistral
    import os
    from dotenv import load_dotenv

    load_dotenv(".env");
    return Mistral, load_dotenv, mo, os


@app.cell
def _(Mistral, os):
    api_key = os.environ["MISTRAL_API_KEY"]

    client = Mistral(api_key=api_key)
    return api_key, client


@app.cell
def _(mo):
    file_selector = mo.ui.file()
    file_selector
    return (file_selector,)


@app.cell
def _(client, file_selector, mo):
    mo.stop(len(file_selector.value) == 0, "Please upload a file")

    uploaded_pdf = client.files.upload(
        file={
            "file_name": "uploaded_file.pdf",
            "content": file_selector.value[0].contents,
        },
        purpose="ocr"
    )
    return (uploaded_pdf,)


@app.cell
def _(client, uploaded_pdf):
    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
    return (signed_url,)


@app.cell
def _(client, signed_url):
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        include_image_base64=True,
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        }
    )
    return (ocr_response,)


@app.cell
def _(mo, ocr_response):
    import json
    from parse import compile as parse_compile

    def page_to_mo(page):
        json_blob = json.loads(page.model_dump_json())
        return replace_img(json_blob)["markdown"]

    def replace_img(json_blob):
        p = parse_compile("![{img_id1}]({img_id2})")
        names = [i['img_id1'] for i in p.findall(json_blob["markdown"]) if i['img_id1'] == i['img_id2']]
        for name in names:
            for img in json_blob["images"]:
                if name in img["id"]: 
                    img_ref = f'<img src="{img['image_base64']}"/>'
                    md_ref = f"![{name}]({name})"
                    json_blob["markdown"] = json_blob["markdown"].replace(md_ref, img_ref)
        return json_blob

    mo.tabs({
        f"page-{i}": page for i, page in enumerate([
            mo.md(page_to_mo(page)) for page in ocr_response.pages
        ])
    })
    return json, page_to_mo, parse_compile, replace_img


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
