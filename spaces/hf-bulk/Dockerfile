FROM python:3.12
COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv

RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN uv pip install -r requirements.txt

COPY --chown=user . /app
USER user

CMD ["marimo", "run", "--sandbox", "app.py", "--include-code", "--host", "0.0.0.0", "--port", "7860"]
