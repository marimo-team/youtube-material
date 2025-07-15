# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==5.5.0",
#     "anthropic==0.56.0",
#     "gitpython==3.1.44",
#     "lancedb==0.24.0",
#     "marimo",
#     "model2vec==0.6.0",
#     "numpy==2.3.1",
#     "polars==1.31.0",
#     "scikit-learn==1.7.0",
#     "srsly==2.5.1",
#     "wigglystuff==0.1.15",
# ]
# ///

import marimo

__generated_with = "0.14.10"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Indexes and LanceDB

    One of the more unique features of [LanceDB](https://www.lancedb.com/documentation/guides/indexing/vector-index.html) is that it allows you to pick how to index our vectors. You can go for the IVF approach which revolves around a technique that doesn't hog the memory. Alternatively, you can for for an HNSW variant that theoretically should be more accurate, but does require more memory at runtime. 

    The whole point of this notebook is to allow you to compare two indices very quickly.
    """
    )
    return


@app.cell(hide_code=True)
def _(file_generator, file_to_items, mo, model_vec, random, table):
    for _ in mo.status.progress_bar(
        range(250),
        title="Adding data to DB",
        subtitle="Working!",
        show_eta=True,
        show_rate=True,
    ):
        # Grab the next file
        try:
            file = next(file_generator)
        except StopIteration:
            continue

        # Process each file's items in batches to ensure consistent lengths
        items = list(file_to_items(file, model=model_vec))
        if len(items):
            table.add(items)
        update = random.random()

    print(f"The table now has {len(table)} items in it.")
    return (update,)


@app.cell(hide_code=True)
def _(alt, dists1, dists2, pl, update):
    import numpy as np

    update

    pltr = (
        pl.concat(
            [dists1.with_columns(kind=pl.lit("hnsw")), dists2.with_columns(kind=pl.lit("ivf-pq"))]
        )
        .select("query_index", "kind", "_distance")
        .with_columns(pl.int_range(pl.len()).over("query_index", "kind").alias("row_idx"))
        .pivot(values="_distance", index=["query_index", "row_idx"], on="kind")
    )

    line_df = pl.DataFrame(
        {
            "x": np.linspace(0, pltr.max()["hnsw"][0], 100),
            "y": np.linspace(0, pltr.max()["hnsw"][0], 100),
        }
    )

    p1 = alt.Chart(pltr).mark_point().encode(x="hnsw", y="ivf-pq")
    p2 = alt.Chart(line_df).mark_line(color="red").encode(x="x", y="y")

    title = f"{pltr.with_columns(p=pl.col('hnsw') > pl.col('ivf-pq')).mean()['p'][0] * 100:.1f}% of scores hnsw > ivf-pq"

    (p1 + p2).properties(title=title)
    return


@app.cell(hide_code=True)
def _(file_generator, file_to_items, lancedb, model_vec):
    db = lancedb.connect(
        uri="lance-demo/database",
    )

    db.drop_all_tables()

    table = db.create_table(
        "benchmark",
        data=list(file_to_items(next(file_generator), model=model_vec)),
        mode="overwrite",
    )
    return (table,)


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Retreival 

    This column contains a text input so you can query the vector database. It contains sentences from abstracts of arxiv articles.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    query_input = mo.ui.text_area(label="input for table")
    query_input
    return (query_input,)


@app.cell(hide_code=True)
def _(model_vec, query_input, table):
    query_vec = model_vec.encode(query_input.value)

    [_["sentence"] for _ in table.search(query_vec).limit(10).select(["sentence"]).to_list()]
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Comparison calculations

    This column calculates the two indices that we want to compare and it also prepares a dataset that we can use to compare numbers.

    If you want to make a change to the indices that you want to compare, you'd do that here.
    """
    )
    return


@app.cell
def _(cs_queries, model_vec, table):
    import polars as pl
    import altair as alt

    table.create_index(metric="cosine", vector_column_name="vector", index_type="IVF_HNSW_SQ")
    table.wait_for_index(["vector_idx"])
    print(table.index_stats("vector_idx"))

    dists1 = table.search(model_vec.encode(cs_queries)).limit(10).to_polars()
    return alt, dists1, pl


@app.cell
def _(cs_queries, model_vec, table):
    table.create_index(metric="cosine", vector_column_name="vector", index_type="IVF_PQ")
    table.wait_for_index(["vector_idx"])
    print(table.index_stats("vector_idx"))

    dists2 = table.search(model_vec.encode(cs_queries)).limit(10).to_polars()
    return (dists2,)


@app.cell(hide_code=True)
def _():
    # 100 queries for computer science arXiv articles
    cs_queries = [
        # Machine Learning & AI
        "deep learning neural networks",
        "transformer architecture attention mechanism",
        "reinforcement learning policy gradient",
        "generative adversarial networks GANs",
        "few-shot learning meta-learning",
        "transfer learning domain adaptation",
        "unsupervised representation learning",
        "self-supervised learning contrastive",
        "federated learning privacy",
        "adversarial examples robustness",
        "explainable AI interpretability",
        "neural architecture search AutoML",
        "graph neural networks GNNs",
        "continual learning catastrophic forgetting",
        "multi-modal learning vision language",
        # Computer Vision
        "object detection YOLO",
        "image segmentation semantic instance",
        "face recognition biometric authentication",
        "optical character recognition OCR",
        "image generation diffusion models",
        "3D reconstruction computer vision",
        "medical image analysis radiology",
        "autonomous driving perception",
        "video understanding temporal modeling",
        "super resolution image enhancement",
        "style transfer neural networks",
        "pose estimation human body",
        "scene understanding indoor outdoor",
        "visual tracking object detection",
        "image classification convolutional networks",
        # Natural Language Processing
        "large language models LLMs",
        "machine translation neural networks",
        "text summarization abstractive extractive",
        "sentiment analysis opinion mining",
        "question answering systems",
        "named entity recognition NER",
        "dialogue systems conversational AI",
        "information extraction text mining",
        "language modeling BERT GPT",
        "text generation natural language",
        "speech recognition automatic transcription",
        "text-to-speech synthesis",
        "multilingual processing cross-lingual",
        "knowledge graphs text understanding",
        "semantic similarity text matching",
        # Systems & Hardware
        "parallel computing distributed systems",
        "GPU computing CUDA optimization",
        "memory management operating systems",
        "compiler optimization code generation",
        "database query optimization",
        "cloud computing virtualization",
        "edge computing IoT devices",
        "high-performance computing HPC",
        "computer architecture processor design",
        "storage systems file systems",
        "network protocols communication",
        "real-time systems embedded computing",
        "performance analysis benchmarking",
        "energy-efficient computing green IT",
        "quantum computing algorithms",
        # Algorithms & Theory
        "sorting algorithms complexity analysis",
        "graph algorithms shortest path",
        "dynamic programming optimization",
        "approximation algorithms NP-hard",
        "randomized algorithms probability",
        "computational complexity theory",
        "algorithm design paradigms",
        "data structures trees graphs",
        "linear algebra numerical methods",
        "combinatorial optimization",
        "game theory algorithmic mechanism",
        "streaming algorithms big data",
        "online algorithms competitive analysis",
        "parallel algorithms synchronization",
        "geometric algorithms computational geometry",
        # Security & Cryptography
        "cybersecurity threat detection",
        "cryptographic protocols blockchain",
        "network security intrusion detection",
        "privacy-preserving computation",
        "secure multi-party computation",
        "homomorphic encryption privacy",
        "digital signatures authentication",
        "malware analysis detection",
        "web security vulnerabilities",
        "mobile security android iOS",
        "zero-knowledge proofs cryptography",
        "secure coding practices",
        "penetration testing ethical hacking",
        "biometric security fingerprint face",
        "IoT security embedded devices",
        # Software Engineering
        "software testing automated testing",
        "code review static analysis",
        "software architecture design patterns",
        "version control git collaboration",
        "continuous integration deployment",
        "software metrics quality assessment",
        "requirements engineering specification",
        "agile development methodologies",
        "software maintenance refactoring",
        "program analysis bug detection",
        "software verification formal methods",
        "API design web services",
        "microservices architecture scalability",
        "software performance optimization",
        "open source software development",
        # Data Science & Analytics
        "big data processing MapReduce Spark",
        "time series analysis forecasting",
        "recommendation systems collaborative filtering",
        "anomaly detection outlier analysis",
        "clustering algorithms unsupervised learning",
        "dimensionality reduction PCA t-SNE",
        "statistical learning hypothesis testing",
        "causal inference observational data",
        "network analysis social networks",
        "information retrieval search engines",
        "data visualization interactive graphics",
        "streaming data real-time analytics",
        "A/B testing experimental design",
        "feature selection machine learning",
        "data mining pattern discovery",
    ]
    return (cs_queries,)


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix 

    This column contains boilerplate for the rest. If you want to change the kind of embedding you can do so below.
    """
    )
    return


@app.cell
def _(StaticModel):
    model_vec = StaticModel.from_pretrained("minishlab/potion-base-8M")
    return (model_vec,)


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from model2vec import StaticModel
    import git


    def clone_repo_gitpython(repo_url, destination):
        """Clone using GitPython library"""
        try:
            # Clone the repository
            repo = git.Repo.clone_from(repo_url, destination)
            print(f"✅ Successfully cloned: {repo_url}")
            print(f"📁 Location: {destination}")
            print(f"📊 Current branch: {repo.active_branch}")
            print(f"📝 Latest commit: {repo.head.commit.hexsha[:8]}")
            return repo

        except git.exc.GitError as e:
            print(f"❌ Git error: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None

    if not Path("input-data").exists():
        repo_url = "https://github.com/koaning/arxiv-frontpage"
        repo = clone_repo_gitpython(repo_url, "input-data")
    return Path, StaticModel, mo


@app.cell
def _():
    import srsly
    import hashlib

    cache = {}


    def md5_hash(text):
        # Convert to bytes if it's a string
        if isinstance(text, str):
            text = text.encode("utf-8")

        # Create MD5 hash
        md5_hasher = hashlib.md5()
        md5_hasher.update(text)

        # Return the hexadecimal digest
        return md5_hasher.hexdigest()


    def file_to_items(path, model, model_name="minishlab/potion-base-8M"):
        urls = []
        for item in srsly.read_jsonl(path):
            embs = model.encode(item["sentences"])
            if (model_name, item["url"]) in cache:
                continue
            cache[(model_name, item["url"])] = item
            for sent, emb in zip(item["sentences"], embs):
                yield {
                    "hash": md5_hash(sent),
                    "sentence": sent,
                    "vector": emb,
                    "meta": {"url": item["url"]},
                }
    return (file_to_items,)


@app.cell
def _(Path):
    file_generator = Path("input-data/data/downloads").glob("*.jsonl")
    return (file_generator,)


@app.cell
def _():
    import lancedb
    import random
    return lancedb, random


@app.cell
def _(table):
    table.schema
    return


if __name__ == "__main__":
    app.run()
