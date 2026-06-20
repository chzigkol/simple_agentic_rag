# Newcomer Guide

This guide gives a recommended path for consuming the repo and getting the most
out of the project.

## Best Path

1. Start with `README.md`

   Read the project purpose, use-case flow, supported routes, repository layout,
   and common commands. The key mental model is:

   ```text
   query -> router -> retrieval or web search -> context -> answer
   ```

2. Read the theory docs before the notebooks

   The docs explain the reasoning behind each notebook. Read them in order:

   ```text
   docs/01_data_and_chroma_sanity_check_theory.md
   docs/02_router_evaluation_theory.md
   docs/03_retrieval_evaluation_theory.md
   docs/04_router_retrieval_cascade_theory.md
   docs/05_relevance_and_web_fallback_theory.md
   docs/06_answer_quality_deepeval_theory.md
   docs/07_full_agentic_rag_benchmark_theory.md
   docs/08_experiments_and_ablation_theory.md
   ```

3. Set up the environment and run tests

   ```bash
   uv sync --extra dev
   uv run pytest
   ```

   You can also run one test tier at a time:

   ```bash
   uv run pytest tests/unit
   uv run pytest tests/integration
   uv run pytest tests/e2e
   ```

4. Inspect the datasets

   Start with:

   ```text
   datasets/medical_qna_dataset.csv
   datasets/medical_device_manuals_dataset.csv
   datasets/evaluation_dataset.csv
   datasets/challenging_router_evaluation_dataset.csv
   ```

   The important idea is that evaluation quality depends on clear examples,
   expected routes, expected answers, and expected document IDs where available.

5. Read the core code in this order

   ```text
   src/agentic_rag/constants.py
   src/agentic_rag/router.py
   src/agentic_rag/ingestion.py
   src/agentic_rag/retrievers.py
   src/agentic_rag/pipeline.py
   src/agentic_rag/evaluation.py
   ```

   This order moves from labels, to decisions, to data loading, to retrieval, to
   the full pipeline, and finally to measurement.

6. Run the notebooks in order

   ```text
   notebooks/01_data_and_chroma_sanity_check.ipynb
   notebooks/02_router_evaluation.ipynb
   notebooks/03_retrieval_evaluation.ipynb
   notebooks/04_router_retrieval_cascade.ipynb
   notebooks/05_relevance_and_web_fallback.ipynb
   notebooks/06_answer_quality_deepeval.ipynb
   notebooks/07_full_agentic_rag_benchmark.ipynb
   notebooks/08_experiments_and_ablation.ipynb
   ```

   `notebooks/00_simple_agentic_rag.ipynb` is useful as a quick demo, but
   notebook `01` is the better serious starting point.

7. Use the CLI after understanding the notebooks

   ```bash
   uv run agentic-rag-evaluate --dataset datasets/evaluation_dataset.csv
   ```

   Then compare behavior on the harder router dataset:

   ```bash
   uv run agentic-rag-evaluate \
     --dataset datasets/challenging_router_evaluation_dataset.csv
   ```

8. Finish with traces

   Once the pipeline makes sense, enable tracing to inspect stage-level behavior:

   ```bash
   OTEL_TRACING_ENABLED=true \
   OTEL_TRACES_EXPORTER=file \
   OTEL_TRACES_FILE=otel_traces/agentic_rag_traces.jsonl \
   uv run agentic-rag-evaluate --dataset datasets/evaluation_dataset.csv --max-examples 1
   ```

## Learning Arc

The intended learning arc is:

```text
understand the task
-> inspect the data
-> evaluate one component
-> connect components
-> benchmark the whole system
-> inspect failures
-> improve deliberately
```
