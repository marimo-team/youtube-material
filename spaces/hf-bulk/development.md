# Development

## Testing your Dockerfile locally

```bash
docker build -t marimo-app .
docker run -it --rm -p 7860:7860 marimo-app
```
