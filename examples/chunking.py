# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anthropic==0.51.0",
#     "jinja2==3.1.6",
#     "marimo",
#     "numpy==2.2.5",
#     "scikit-learn==1.6.1",
#     "sentence-transformers==4.1.0",
#     "transformers==4.51.3",
# ]
# ///

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer, mo


@app.cell
def _():
    from jinja2 import Template
    return (Template,)


@app.cell
def _(SentenceTransformer):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return


@app.cell
def _():
    from transformers import AutoModel, AutoTokenizer
    import numpy as np

    jina_tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    jina_model     = AutoModel.from_pretrained(    'jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    return jina_model, jina_tokenizer, np


@app.cell
def _(create_token_visualization_template):
    def visualize_tokens(tokens, values):
        """
        Renders tokens with background colors based on their values.

        Args:
            tokens (list): List of tokens to visualize.
            values (list): Corresponding numeric values for each token.

        Returns:
            HTML visualization of the tokens.
        """
        if len(tokens) != len(values):
            raise ValueError("The number of tokens must match the number of values")

        if len(tokens) == 0:
            return "No tokens to visualize"

        template = create_token_visualization_template()
        min_value = min(values)
        max_value = max(values)

        html_content = template.render(
            tokens=tokens,
            values=values,
            min_value=min_value,
            max_value=max_value
        )

        return html_content
    return (visualize_tokens,)


@app.cell
def _(Template):
    def create_token_visualization_template():
        """
        Creates a Jinja2 template for visualizing tokens with background colors
        based on their associated numeric values.
    
        The template uses a gray-to-red color scheme that ensures good contrast with white text
        and makes it easier to interpret the intensity of values.

        Returns:
            Template: A Jinja2 template object.
        """
        template_string = """
        <style>
        .token-container {
            font-family: 'Arial', sans-serif;
            line-height: 2;
        }
        .token {
            padding: 3px 6px;
            border-radius: 4px;
            display: inline-block;
            margin: 2px;
            font-size: 14px;
            color: white;
            text-shadow: 0px 0px 2px rgba(0,0,0,0.6);
        }
        .legend {
            margin-top: 10px;
            display: flex;
            align-items: center;
            font-size: 12px;
        }
        .gradient {
            height: 20px;
            width: 200px;
            background: linear-gradient(to right, rgb(100,100,100), rgb(220,30,30));
            margin: 0 10px;
            border-radius: 2px;
        }
        </style>

        <div class="token-container">
        {% for i in range(tokens|length) %}
            {% set token = tokens[i] %}
            {% set value = values[i] %}
            {% set intensity = (value - min_value) / (max_value - min_value) if max_value != min_value else 0.5 %}
        
            {# Calculate color components for gray-to-red gradient #}
            {% set base_gray = 100 %}
            {% set red = (base_gray + (220 - base_gray) * intensity)|round|int %}
            {% set green = (base_gray - (base_gray - 30) * intensity)|round|int %}
            {% set blue = (base_gray - (base_gray - 30) * intensity)|round|int %}
        
            {% set color = 'rgb(' + red|string + ',' + green|string + ',' + blue|string + ')' %}
        
            <span class="token" style="background-color: {{ color }};">{{ token }}</span>
        {% endfor %}

        <div class="legend">
            <span>{{ min_value|round(2) }}</span>
            <div class="gradient"></div>
            <span>{{ max_value|round(2) }}</span>
        </div>
        </div>
        """
        return Template(template_string)
    return (create_token_visualization_template,)


@app.cell
def _():
    return


@app.cell(column=1)
def _(mo):
    text_widget = mo.ui.text_area(label="input text").form()
    query_widget = mo.ui.text_area(label="input text").form()
    return query_widget, text_widget


@app.cell
def _(text_widget):
    text_widget
    return


@app.cell
def _(query_widget):
    query_widget
    return


@app.cell
def _(base_embed, mo, query_widget, similarity, text_widget, visualize_tokens):
    _tokens, _vecs = base_embed(text_widget.value, kind="tokens")
    _q_vec = base_embed(query_widget.value)

    mo.Html(visualize_tokens(_tokens, similarity(_vecs, _q_vec)))
    return


@app.cell
def _(base_embed, mo, query_widget, similarity, text_widget, visualize_tokens):
    def chunked_emb(s):
        _tokens, _vecs = base_embed(s, kind="tokens")
        _q_vec = base_embed(query_widget.value)
        _sims = [float(_) for _ in similarity(_vecs, _q_vec)]
        return visualize_tokens(_tokens, _sims)

    mo.Html("<br>".join([chunked_emb(s) for s in text_widget.value.split(". ")]))
    return


@app.cell
def _(np):
    def similarity(v, v2):
        return 1 - (np.dot(v, v2) / (np.linalg.norm(v) * np.linalg.norm(v2)))
    return (similarity,)


@app.cell
def _():
    return


@app.cell(column=2)
def _(jina_tokenizer):
    punctuation_mark_id = jina_tokenizer.convert_tokens_to_ids('.')
    return


@app.cell
def _(jina_model, jina_tokenizer):
    def base_embed(query, kind="pooled"): 
        inputs = jina_tokenizer(query, return_tensors='pt')
        if kind == "pooled":
            return jina_model(**inputs)[0].detach().numpy()[0].mean(axis=0)
        if kind == "tokens":
            inputs_tok = jina_tokenizer(query, return_tensors='pt', return_offsets_mapping=True)
            m = inputs_tok["offset_mapping"]
            out = [query[_[0]: _[1]] for _ in m[0]]
            return out, jina_model(**inputs)[0].detach().numpy()[0]
    return (base_embed,)


@app.function(hide_code=True)
def late_chunking(document, model, tokenizer):
    "Implements late chunking on a document."

    # Tokenize with offset mapping to find sentence boundaries
    inputs_with_offsets = tokenizer(document, return_tensors='pt', return_offsets_mapping=True)
    token_offsets = inputs_with_offsets['offset_mapping'][0]
    token_ids = inputs_with_offsets['input_ids'][0]
    
    # Find chunk boundaries
    punctuation_mark_id = tokenizer.convert_tokens_to_ids('.')    
    chunk_positions, token_span_annotations = [], []
    span_start_char, span_start_token = 0, 0

    for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets)):
        if i < len(token_ids)-1:
            if token_id == punctuation_mark_id and document[end:end+1] in [' ', '\n']:
                # Store both character positions and token positions
                chunk_positions.append((span_start_char, int(end)))
                token_span_annotations.append((span_start_token, i+1))
                
                # Update start positions for next chunk
                span_start_char, span_start_token = int(end)+1, i+1
    
    # Create text chunks from character positions
    chunks = [document[start:end].strip() for start, end in chunk_positions]
    
    # Encode the entire document
    inputs = tokenizer(document, return_tensors='pt')
    model_output = model(**inputs)
    token_embeddings = model_output[0]
    
    # Create embeddings for each chunk using mean pooling
    embeddings = []
    for start_token, end_token in token_span_annotations:
        if end_token > start_token:  # Ensure span has at least one token
            chunk_embedding = token_embeddings[0, start_token:end_token].mean(dim=0)
            embeddings.append(chunk_embedding.detach().cpu().numpy())
    
    return chunks, embeddings


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Sentence chunking""")
    return


@app.cell
def _(base_embed, cosine_similarity, docs, query_widget):
    _sims = cosine_similarity([base_embed(_) for _ in docs], [base_embed(query_widget.value)])

    for _d, _s in sorted(zip(docs, _sims), key=lambda d: -d[1]):
        print(_s, _d)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Late chunking""")
    return


@app.cell
def _(base_embed, jina_model, jina_tokenizer, query_widget, text_widget):
    from sklearn.metrics.pairwise import cosine_similarity

    docs, vecs = late_chunking(text_widget.value, jina_model, jina_tokenizer)
    _sims = cosine_similarity(vecs, [base_embed(query_widget.value)])

    for _d, _s in sorted(zip(docs, _sims), key=lambda d: -d[1]):
        print(_s, _d)
    return cosine_similarity, docs


@app.cell(column=3)
def _():
    return


@app.cell(column=4)
def _():
    return


if __name__ == "__main__":
    app.run()
