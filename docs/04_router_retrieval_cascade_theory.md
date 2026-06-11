# 04 - Router Retrieval Cascade Theory

Cascade evaluation checks what happens when routing and retrieval are connected.

## Core Idea

A good retriever can still fail if the router sends the query to the wrong source. This notebook measures how upstream routing decisions affect downstream retrieval quality.

## Two Retrieval Views

- Expected-source retrieval: search the correct source from the label
- Predicted-source retrieval: search the source chosen by the router

Expected-source retrieval is the control. Predicted-source retrieval shows the real system behavior.

## What This Notebook Measures

- Router correctness
- Retrieval quality using the expected source
- Retrieval quality using the predicted source
- Route-induced retrieval failures
- Cases where routing and retrieval disagree

## Main Lesson

Do not debug retrieval failures blindly. First check whether retrieval searched the right collection. In multi-step systems, failures can cascade from one stage into the next.
