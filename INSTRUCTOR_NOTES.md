# Instructor Notes

These notes follow `NEWCOMER_GUIDE.md` and are meant as concise prompts for
teaching the repo walkthrough.

## 1. Start With `README.md`

Set the frame: this is not just a RAG demo, it is an evaluation-first project.
The audience should understand the core flow before reading code:

```text
query -> router -> retrieval or web search -> context -> answer
```

Emphasize that the repo is intentionally small so each system decision can be
seen, measured, and debugged.

## 2. Read The Theory Docs Before The Notebooks

Explain that the docs are the "why" and the notebooks are the "how".

The main teaching point is sequencing: do not jump directly to answer quality.
First verify data, routing, retrieval, cascade behavior, and only then
end-to-end quality.

Useful phrase:

```text
End-to-end metrics summarize behavior; component metrics explain behavior.
```

## 3. Set Up The Environment And Run Tests

Use the tests to introduce engineering discipline around AI systems.

The three tiers mean:

- `unit`: isolated behavior, fast feedback.
- `integration`: real components connected, such as Chroma ingestion/retrieval.
- `e2e`: the full local pipeline path.

Stress that deterministic tests are especially valuable in AI projects because
they give a stable baseline before model variability enters.

## 4. Inspect The Datasets

Make the audience slow down here. The quality of the evaluation depends more on
the dataset design than on the metric code.

There are two kinds of datasets in the repo:

- Source datasets: the local knowledge the system can retrieve from.
- Evaluation datasets: labeled examples used to measure routing, retrieval, and
  answer behavior.

### `medical_qna_dataset.csv`

This is the general medical Q&A knowledge source.

Columns:

- `qtype`: the type of medical question.
  Example: `symptoms`, `treatment`, `frequency`, `susceptibility`.
- `Question`: the source question text.
  Example: `What are the symptoms of Lymphocytic Choriomeningitis (LCM) ?`
- `Answer`: the source answer text that becomes retrievable context.
  Example: an explanation of LCM symptoms and neurological disease.

Teaching point: this dataset supports the `Retrieve_QnA` route. It is good for
stable medical knowledge, not recent news or device-manual details.

### `medical_device_manuals_dataset.csv`

This is the medical-device/manual knowledge source.

Columns:

- `Device_Name`: device family or product name.
  Example: `Insulin Pump`, `Ventilator`.
- `Model_Number`: model identifier.
  Example: `DAN246`, `Model 4428`.
- `Manufacturer`: company name.
  Example: `Danaher`, `Fresenius Medical Care`.
- `Manual_Version`: manual version string.
  Example: `2023-05-C`, `v8.4`.
- `Publication_Date`: manual publication date.
  Example: `2022-01-27`.
- `Device_Class`: regulatory class.
  Example: `Class I`, `Class II`.
- `Regulatory_Approval_ID`: approval or submission identifier.
  Example: `NDA412861`, `IDE380253`.
- `Patient_Population`: intended patient group.
  Example: `Adult`, `Adult and Pediatric`.
- `Indications_for_Use`: when or why the device is intended to be used.
  Example: `Indicated for real-time heart rate assessment...`
- `Contraindications`: when the device should not be used.
  Example: `Not recommended for use in radiation therapy patients...`
- `Sterilization_Method`: sterilization approach, when available.
  Example: `Single-Use Sterile`.
- `Number_of_Warnings`: count of warnings in the manual metadata.
  Example: `11`.
- `Number_of_Cautions`: count of cautions in the manual metadata.
  Example: `13`.
- `Device_Lifetime_Years`: expected lifetime in years.
  Example: `5.0`, `8.0`.
- `Device_Weight_kg`: device weight.
  Example: `7.85`, `69.53`.
- `Max_Operating_Temperature_C`: maximum operating temperature.
  Example: `12.0`, `26.0`.

Teaching point: this dataset supports the `Retrieve_Device` route. It is the
right place for model numbers, contraindications, patient population, manual
metadata, and manufacturer/device-specific questions.

### `evaluation_dataset.csv`

This is the main labeled evaluation set.

Columns:

- `query`: the user-style question being evaluated.
  Example: `What are the treatments for Kawasaki disease ?`
- `expected_source_type`: the correct route label.
  Example: `Retrieve_QnA`, `Retrieve_Device`, `Web_Search`.
- `expected_collection`: the expected local Chroma collection when retrieval is
  local.
  Example: `medical_qna`, `medical_device_manual`.
- `expected_doc_ids`: the expected evidence document IDs, when available.
  Example: `0`, `1`, `2` in the CSV, resolved by code to IDs such as `qna-0`.
- `expected_answer`: the reference answer or answer snippet used for inspection
  and answer-quality checks.
  Example: the Kawasaki disease treatment answer.
- `category`: a slice label for analysis.
  Example: `frequency`, `treatment`, `symptoms`, `genetic changes`.

Teaching point: this dataset is the golden set for the normal benchmark. It lets
the repo evaluate both source selection and evidence retrieval.

### `challenging_router_evaluation_dataset.csv`

This is a harder router-focused dataset.

Columns:

- `query`: a deliberately ambiguous or difficult user question.
  Example: a question that mentions both general medical risk and a specific
  device manual contraindication.
- `expected_source_type`: the correct route despite ambiguity.
  Example: `Retrieve_Device` when the answer depends on a manual.
- `category`: the failure-mode slice.
  Example: `device_plus_general_medical`, `explicit_source_meta_question`.
- `rationale`: why that route is expected.
  Example: `Mentions general risks, but the answer depends on a device manual contraindication.`

Teaching point: easy examples can make a router look solved. Challenging
examples reveal ambiguity, recency wording, source-selection mistakes, and
policy gaps.

Point out during the walkthrough:

- `expected_source_type` is the router label.
- `expected_doc_ids` supports retrieval scoring when gold evidence exists.
- `expected_answer` supports answer-quality checks.
- The challenging router dataset is meant to expose ambiguity and edge cases.

Teaching message:

```text
Bad labels produce confident but misleading metrics.
```

## 5. Read The Core Code In Order

Use this as the code-reading path:

```text
constants -> router -> ingestion -> retriever -> pipeline -> evaluation
```

Explain what each layer owns:

- `constants.py`: the vocabulary of decisions.
- `router.py`: where the system chooses a source.
- `ingestion.py`: how raw CSV rows become searchable documents.
- `retrievers.py`: how local evidence is fetched.
- `pipeline.py`: how the full answer path is assembled.
- `evaluation.py`: how behavior becomes measurable.

The important point is separation of concerns: each file maps to a system stage
that can be tested and evaluated separately.

## 6. Run The Notebooks In Order

Present the notebooks as an evaluation ladder, not a random tutorial set.

Recommended framing:

```text
01 proves the substrate is sane.
02 checks the first decision.
03 checks evidence retrieval.
04 shows cascading failures.
05 looks at answer usefulness.
06 summarizes the full path.
07 compares variants and tradeoffs.
```

Remind the audience that notebook `00` is a quick demo, but notebook `01` is
where serious evaluation starts.

## 7. Use The CLI After Understanding The Notebooks

The CLI is the operational version of the notebook ideas.

Explain that notebooks help with exploration and debugging, while the CLI makes
the evaluation repeatable.

Key distinction:

```text
Notebook = learn and inspect.
CLI = repeat and compare.
```

Encourage comparing the base and challenging datasets to show how aggregate
performance changes when examples become more realistic.

## 8. Finish With Traces

Use tracing to connect metrics back to runtime behavior.

The audience should understand that traces answer questions like:

- Which route did the system choose?
- How many documents were retrieved?
- Which stage took time?
- Did the answer path use local context or fallback?

Teaching message:

```text
Metrics tell you what happened; traces help show where it happened.
```

## Closing Message

End by reinforcing the intended learning arc:

```text
understand the task
-> inspect the data
-> evaluate one component
-> connect components
-> benchmark the whole system
-> inspect failures
-> improve deliberately
```

The repo's main lesson is that AI quality becomes manageable when the system is
decomposed into observable, testable, and evaluable stages.
