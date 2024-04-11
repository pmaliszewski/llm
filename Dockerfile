FROM python:3.11-slim as base
WORKDIR /usr/src/app
ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app"
RUN pip install --no-cache-dir poetry
RUN poetry config virtualenvs.create false
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi
FROM base as final
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "ui.py"]
