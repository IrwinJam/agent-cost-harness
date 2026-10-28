# Run the harness without installing Python.
#
#   docker build -t agent-cost-harness .
#   docker run --rm -p 8787:8787 agent-cost-harness
#   docker run --rm -p 8787:8787 -e HARNESS_API_KEY -v "$PWD/runs:/app/runs" \
#     agent-cost-harness serve --base-url https://api.example.com/v1 --provider myprovider --policies cache
#
# Inside the container the server listens on all interfaces so Docker can forward the port. Only
# publish it to your own machine, like the -p flags above do.

FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY harness ./harness
RUN pip install --no-cache-dir .

ENV HARNESS_HOST=0.0.0.0
EXPOSE 8787
ENTRYPOINT ["python", "-m", "harness"]
CMD ["serve", "--simulate"]
