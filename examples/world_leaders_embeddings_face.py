# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==5.5.0",
#     "facenet-pytorch==2.5.3",
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.2.6",
#     "pandas==2.2.3",
#     "pillow==11.2.1",
#     "scikit-learn==1.6.1",
#     "scipy==1.15.3",
#     "torch==2.7.0",
#     "torchvision==0.22.0",
#     "umap-learn==0.5.7",
# ]
# ///

import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Face Embeddings of World Leaders

        This notebook explores face embeddings using a subset of the **Labeled Faces in the Wild** dataset, focused on public figures. We'll use standard Python and scikit-learn libraries to load the data, embed images, reduce dimensionality, and visualize clustering behavior.

        This example builds on a demo from the Marimo gallery using the MNIST dataset. Here, we adapt it to work with a facial recognition dataset of public figures. While facial recognition has limited responsible use cases, this curated subset includes only world leaders — a group I feel comfortable experimenting with in a technical context.

        We'll start with our imports:
        """
    )
    return


@app.cell
def _():
    from time import time

    import matplotlib.pyplot as plt
    from scipy.stats import loguniform

    from sklearn.datasets import fetch_lfw_people
    from sklearn.decomposition import PCA
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report
    from sklearn.model_selection import RandomizedSearchCV, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return (
        ConfusionMatrixDisplay,
        PCA,
        RandomizedSearchCV,
        SVC,
        StandardScaler,
        classification_report,
        fetch_lfw_people,
        loguniform,
        plt,
        time,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We're using `fetch_lfw_people` from `sklearn.datasets` to load a curated subset of the LFW dataset — restricted to individuals with at least 70 images, resulting in 7 distinct people and just over 1,200 samples. These happen to be mostly world leaders, which makes the demo both manageable and fun to explore.""")
    return


@app.cell
def _(fetch_lfw_people):
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    # introspect the images arrays to find the shapes (for plotting)
    n_samples, h, w = lfw_people.images.shape

    # for machine learning we use the 2 data directly (as relative pixel
    # positions info is ignored by this model)
    X = lfw_people.data
    n_features = X.shape[1]

    # the label to predict is the id of the person
    Y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)
    return (
        X,
        Y,
        h,
        lfw_people,
        n_classes,
        n_features,
        n_samples,
        target_names,
        w,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Next, we embed each face image using a pre-trained FaceNet model (`InceptionResnetV1` trained on `vggface2`). This converts each image into a 512-dimensional vector. Since the original data is grayscale and flattened, we reshape, normalize, and convert it to RGB before feeding it through the model.""")
    return


@app.cell
def _(X, h, w):
    from facenet_pytorch import InceptionResnetV1
    from torchvision import transforms
    from PIL import Image
    import torch
    import numpy as np

    # Load FaceNet model
    model = InceptionResnetV1(pretrained='vggface2').eval()

    # Transform pipeline: grayscale → RGB → resize → normalize
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize([0.5], [0.5])
    ])

    # Embed a single flattened row from X
    def embed_flat_row(flat):
        img = flat.reshape(h, w)
        img = (img * 255).astype(np.uint8)
        pil = Image.fromarray(img).convert("L")  # grayscale
        tensor = transform(pil).unsqueeze(0)
        with torch.no_grad():
            return model(tensor).squeeze().numpy()  # 512-dim

    # Generate embeddings for all samples
    embeddings = np.array([embed_flat_row(row) for row in X])
    return (
        Image,
        InceptionResnetV1,
        embed_flat_row,
        embeddings,
        model,
        np,
        torch,
        transform,
        transforms,
    )


@app.cell
def _(mo):
    mo.md(r"""Now that we have 512-dimensional embeddings, we reduce them to 2D for visualization. Both t-SNE and UMAP are available here — UMAP is active by default, but you can switch to t-SNE by uncommenting the alternate line. This step lets us inspect the structure of the embedding space:""")
    return


@app.cell
def _(embeddings):
    from sklearn.manifold import TSNE
    import umap.umap_ as umap

    # X_embedded = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)
    X_embedded = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
    return TSNE, X_embedded, umap


@app.cell
def _(mo):
    mo.md(r"""We wrap the 2D embeddings into a Pandas DataFrame for easier manipulation and plotting. Each row includes x/y coordinates and the associated person ID, which we map to names. We then define a simple Altair scatterplot function to visualize the clustered embeddings by identity.""")
    return


@app.cell
def _(X_embedded, Y, target_names):
    import pandas as pd

    embedding_df = pd.DataFrame({
        "x": X_embedded[:, 0],
        "y": X_embedded[:, 1],
        "person": Y
    }).reset_index()
    embedding_df["name"] = embedding_df["person"].map(lambda i: target_names[i])
    return embedding_df, pd


@app.cell
def _():
    import altair as alt
    def scatter(df):
        return (alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X("x:Q"),
            y=alt.Y("y:Q"),
            color=alt.Color("name:N"),
        ).properties(width=500, height=500))
    return alt, scatter


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here's our 2D embedding space of world leader faces! Each point is a facial embedding projected with UMAP and colored by identity. Try selecting a cluster — the notebook will automatically reveal the associated images so you can explore what the model “thinks” belongs together.""")
    return


@app.cell
def _(embedding_df, scatter):
    import marimo as mo
    chart = mo.ui.altair_chart(scatter(embedding_df))
    chart
    return chart, mo


app._unparsable_cell(
    r"""
    When you select points in the scatterplot, Marimo automatically passes those indices into this cell. Here, we render a preview of the corresponding face images using `matplotlib`, along with a table of all selected metadata — making it easy to inspect clustering quality or outliers at a glance.
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


@app.cell
def _(chart, mo):
    table = mo.ui.table(chart.value)
    return (table,)


@app.cell
def _(X, chart, h, mo, table, w):
    # show 10 images: either the first 10 from the selection, or the first ten
    # selected in the table
    mo.stop(not len(chart.value))

    def show_images(indices, max_images=10):
        import matplotlib.pyplot as plt

        indices = indices[:max_images]
        images = X.reshape((-1, h, w))[indices]
        fig, axes = plt.subplots(1, len(indices))
        fig.set_size_inches(12.5, 1.5)
        if len(indices) > 1:
            for im, ax in zip(images, axes.flat):
                ax.imshow(im, cmap="gray")
                ax.set_yticks([])
                ax.set_xticks([])
        else:
            axes.imshow(images[0], cmap="gray")
            axes.set_yticks([])
            axes.set_xticks([])
        plt.tight_layout()
        return fig

    selected_images = (
        show_images(list(chart.value["index"]))
        if not len(table.value)
        else show_images(list(table.value["index"]))
    )

    mo.md(
        f"""
        **Here's a preview of the images you've selected**:

        {mo.as_html(selected_images)}

        Here's all the data you've selected.

        {table}
        """
    )
    return selected_images, show_images


if __name__ == "__main__":
    app.run()
