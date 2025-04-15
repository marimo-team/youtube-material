import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from smartfunc import backend, get_backend_models
    from flowshow import task
    from diskcache import Cache
    from dotenv import load_dotenv
    import random
    import replicate
    from uuid import uuid4

    load_dotenv(".env")
    return (
        Cache,
        backend,
        get_backend_models,
        load_dotenv,
        mo,
        random,
        replicate,
        task,
        uuid4,
    )


@app.cell
def _(Cache):
    cache = Cache("image-gen")
    titles = ["Meta's Antitrust Trial: Zuckerberg's Tech Empire Under Scrutiny","DIY AI Butler: A Hacker's Guide to Personal Digital Assistance"]
    return cache, titles


@app.cell
def _(backend, cache, colors, replicate, uuid4):
    @backend("gpt-3.5-turbo")
    def image_prompt_segment(title): 
            """This is the title of a newspaper article that needs an image: {{title}}. This needs to be represented visually. I am looking for a single word, maybe two, that could be taken from this title that could be used to generate an image that reflects the theme. The phrase should convey a visual meaning."""


    def run_on_replicate(prompt, model="google/imagen-3"):
        return replicate.run(
            model,
            input={
                "prompt": prompt,
                "aspect_ratio": "16:9",
                "negative_prompt": "logos, text, words, description",
                "safety_filter_level": "block_medium_and_above"
            }
        )

    @cache.memoize()
    def generate_image(title, color, style, model, seed=1):
        segment = image_prompt_segment(title)
    
        image_prompt = f"{segment}, {style}, no text what so ever, attention in the center"
        if color:
            image_prompt += f"a touch of {colors}"
        output = run_on_replicate(image_prompt)
    
        gen_uuid = str(uuid4())
    
        out_path = f'images/{gen_uuid}.webp'

        # Download and convert with specific output path
        converted_image_path = download_and_convert_to_webp(
            output.url,
            output_path=out_path
        )

        return {
            "image": out_path,
            "segment": segment, 
            "prompt": image_prompt, 
            "title": title
        }

    import os
    import requests
    from PIL import Image

    def download_and_convert_to_webp(url, output_path=None):
        try:
            # Download the image
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Create a temporary file to save the downloaded image
            with open('temp_download.png', 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            # Open the downloaded image
            with Image.open('temp_download.png') as img:
                # Determine output path
                if output_path is None:
                    # Extract filename from URL and change extension
                    output_path = os.path.splitext(os.path.basename(url))[0] + '.webp'

                # Convert and save as WebP
                img.save(output_path, 'WEBP',) 
                         # quality=0.8, 
                         # method=6,  # Highest compression method
                         # lossless=False)

            # Remove temporary file
            os.remove('temp_download.png')

            return output_path

        except requests.RequestException as e:
            print(f"Error downloading image: {e}")
            raise
        except IOError as e:
            print(f"Error processing image: {e}")
            # Remove temporary file if it exists
            if os.path.exists('temp_download.png'):
                os.remove('temp_download.png')
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            # Remove temporary file if it exists
            if os.path.exists('temp_download.png'):
                os.remove('temp_download.png')
            raise
    return (
        Image,
        download_and_convert_to_webp,
        generate_image,
        image_prompt_segment,
        os,
        requests,
        run_on_replicate,
    )


@app.cell
def _(mo):
    color = mo.ui.dropdown(
        options=[""],
        value="",
        label="Color"
    )
    style = mo.ui.dropdown(
        options=["pixel techno", "detailed ink illustration", "retro game pixel art"],
        value="pixel techno", 
        label="style"
    )
    model = mo.ui.dropdown(
        options=["google/imagen-3", "google/imagen-3-fast"],
        value="google/imagen-3-fast",
        label="model", 
    )
    return color, model, style


@app.cell
def _(color, mo, model, style, titles):
    form1 = mo.md("""
    ## Settings for image one
      
    {color} {style} {model}
    """).batch(color=color, style=style, model=model).form()

    form2 = mo.md("""
    ## Settings for image two
      
    {color} {style} {model}
    """).batch(color=color, style=style, model=model).form()

    general_form = mo.md("""
    ## General settings

    {n_images}

    {title}
    """).batch(
        n_images=mo.ui.slider(1, 4, 1, label="number of images"), 
        title=mo.ui.dropdown(options=titles, value=titles[0])
    )
    return form1, form2, general_form


@app.cell
def _(form1, form2, general_form, mo):
    mo.vstack([
        general_form, 
        mo.hstack([
            form1, form2
        ])
    ])
    return


@app.cell
def _(form1, form2, mo, show_imgs):
    mo.hstack([
        show_imgs(form1),
        show_imgs(form2)
    ])
    return


@app.cell
def _(form1, form2, general_form, generate_image):
    for i in range(general_form.value["n_images"]): 
        generate_image(general_form.value["title"], **form1.value, seed=i)
        generate_image(general_form.value["title"], **form2.value, seed=i)
    return (i,)


@app.cell
def _(general_form, generate_image, mo):
    def show_imgs(form):
        return mo.vstack([mo.image(generate_image(general_form.value["title"], **form.value, seed=_)["image"]) 
                for _ in range(general_form.value["n_images"])])
    return (show_imgs,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
