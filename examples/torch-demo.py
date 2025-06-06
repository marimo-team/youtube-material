# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==5.5.0",
#     "datasets==3.6.0",
#     "marimo",
#     "matplotlib==3.10.1",
#     "mofresh==0.2.1",
#     "mohtml==0.1.7",
#     "numpy==2.2.5",
#     "polars==1.29.0",
#     "scikit-learn==1.6.1",
#     "sentence-transformers==4.1.0",
#     "torch==2.7.0",
# ]
# ///

import marimo

__generated_with = "0.13.8"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""## Dataset setup""")
    return


@app.cell
def _():
    import marimo as mo
    from datasets import load_dataset

    dataset = load_dataset("m3hrdadfi/recipe_nlg_lite")
    print(dataset)
    return dataset, mo


@app.cell
def _(text_train):
    text_train
    return


@app.cell
def _(dataset):
    text_train, text_test = dataset["train"]["name"], dataset["test"]["name"] 
    return text_test, text_train


@app.cell
def _(text_test, text_train):
    from sentence_transformers import SentenceTransformer

    tfm = SentenceTransformer("all-MiniLM-L6-v2")
    X_train = tfm.encode(text_train)
    X_test = tfm.encode(text_test)
    return X_test, X_train


@app.cell
def _(text_test, text_train):
    import numpy as np

    words_of_interest = "pork potato salad chicken rice".split(" ")

    def to_label(texts, words):
        return np.array([[word in _ for word in words] for _ in texts]).astype(int)

    for word in words_of_interest:
        print(word, np.mean([word in _ for _ in text_train]), np.mean([word in _ for _ in text_test]))

    y_train = to_label(text_train, words_of_interest)
    y_test = to_label(text_test, words_of_interest)
    return np, words_of_interest, y_test, y_train


@app.cell
def _(nn, torch):
    class SimpleFeedForward(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(SimpleFeedForward, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc3 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.relu(out)
            out = self.fc3(out)
            return torch.sigmoid(out)
    return (SimpleFeedForward,)


@app.cell
def _():
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    from mohtml import table, th, tr, td, tailwind_css, p
    import polars as pl
    import altair as alt
    import torch
    import torch.nn as nn
    import torch.optim as optim

    tailwind_css()
    return (
        accuracy_score,
        alt,
        nn,
        optim,
        pl,
        precision_score,
        recall_score,
        torch,
    )


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""## Pytorch Setup""")
    return


@app.cell
def _(
    SimpleFeedForward,
    X_test,
    X_train,
    accuracy_score,
    loss_widget,
    nn,
    optim,
    plot_epoch_loss,
    plot_epoch_precision,
    precision_score,
    recall_score,
    test_precision_widget,
    torch,
    words_of_interest,
    y_test,
    y_train,
):
    X_train_tensor, X_test_tensor = torch.FloatTensor(X_train), torch.FloatTensor(X_test)
    y_train_tensor, y_test_tensor = torch.FloatTensor(y_train), torch.FloatTensor(y_test)

    def performance(x_tensor, y_tensor, epoch, group, threshold=0.5):
        preds = (model(x_tensor) > threshold)
        rows = []

        for i, _word in enumerate(words_of_interest):
            predicted = preds.numpy()[:, i]
            accuracy = accuracy_score(y_tensor[:, i], predicted)
            precision = precision_score(y_tensor[:, i], predicted, average="macro")
            recall = recall_score(y_tensor[:, i], predicted, average="macro")
            rows.append({
                "metric": "precision",
                "word": _word,
                "value": float(precision),
                "epoch": epoch,
                "group": group
            })
        return rows

    learning_rate = 0.0003
    num_epochs = 800

    # Initialize the model
    model = SimpleFeedForward(
        input_size=X_train.shape[1], 
        hidden_size=64, 
        num_classes=y_train.shape[1])

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    loss_data = []
    metric_data = []
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor.float())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_data.append({"epoch": epoch, "loss": loss.item(), "group": "train"})

        loss_valid = criterion(model(X_test_tensor), y_test_tensor.float())
        loss_data.append({
            "epoch": epoch, "loss": loss_valid.item(), "group": "test"
        })
        metric_data.extend(performance(X_test_tensor, y_test_tensor, epoch=epoch, group="train"))
        metric_data.extend(performance(X_train_tensor, y_train_tensor, epoch=epoch, group="test"))

        loss_widget.src = plot_epoch_loss(loss_data)
        test_precision_widget.src = plot_epoch_precision(metric_data, kind="test")
    return


@app.cell
def _(alt, altair2svg, pl):
    def make_svg_chart(metric_data):
        df = pl.DataFrame(metric_data)
        chart = alt.Chart(df).mark_line().encode(x="epoch", y="value", color="word:N", strokeDash="group")
        return altair2svg(chart)
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""## Views""")
    return


@app.cell
def _(loss_widget):
    loss_widget
    return


@app.cell
def _(test_precision_widget):
    test_precision_widget
    return


@app.cell
def _():
    from mofresh import ImageRefreshWidget, HTMLRefreshWidget, refresh_matplotlib, altair2svg
    import matplotlib.pylab as plt

    loss_widget = ImageRefreshWidget()
    test_precision_widget = ImageRefreshWidget()
    return (
        HTMLRefreshWidget,
        ImageRefreshWidget,
        altair2svg,
        loss_widget,
        plt,
        refresh_matplotlib,
        test_precision_widget,
    )


@app.cell
def _(plt, refresh_matplotlib, words_of_interest):
    @refresh_matplotlib
    def plot_epoch_loss(data):
        plt.plot(
            [_["epoch"] for _ in data if _["group"] == "train"], 
            [_["loss"] for _ in data if _["group"] == "train"],
            label="train"
        )
        plt.plot(
            [_["epoch"] for _ in data if _["group"] == "test"], 
            [_["loss"] for _ in data if _["group"] == "test"],
            label="test"
        )
        plt.title("loss over epochs")
        plt.legend()

    @refresh_matplotlib
    def plot_epoch_precision(data, kind="train"):
        for _word in words_of_interest:
            plt.plot(
                [_["epoch"] for _ in data if _["group"] == kind and _["word"] == _word], 
                [_["value"] for _ in data if _["group"] == kind and _["word"] == _word],
                label=_word
            )
        plt.title(f"{kind} precision over epochs")
        plt.legend()
    return plot_epoch_loss, plot_epoch_precision


@app.cell
def _(HTMLRefreshWidget):
    tbl_train = HTMLRefreshWidget()
    tbl_test = HTMLRefreshWidget()
    return


@app.cell(column=3)
def _(ImageRefreshWidget):
    pi_widget = ImageRefreshWidget(src="")
    pi_widget
    return (pi_widget,)


@app.cell
def _(pi_widget, plot_sim):
    import random

    xy = []
    for _i in range(10000):
        xy.append([random.random(), random.random()])
        if _i % 50:
            pi_widget.src = plot_sim(xy)
    return


@app.cell
def _(np, plt, refresh_matplotlib):
    @refresh_matplotlib
    def plot_sim(data):
        x = np.array(data)
        c = np.sqrt(x[:, 0]**2 + x[:, 1]**2) <= 1
        plt.figure(figsize=(5, 5))
        plt.scatter(x[:, 0], x[:, 1], c=c, s=5)
        plt.title(f"{np.mean(c)*100:.2f}% pi = {np.mean(c)*4:.2f}")
    return (plot_sim,)


@app.cell(column=4)
def _():
    return


if __name__ == "__main__":
    app.run()
