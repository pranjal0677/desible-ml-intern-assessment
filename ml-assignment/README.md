# AI/ML Intern Assignment

This repository contains the solutions for the AI/ML Intern Assignment, featuring a Trigram Language Model built from scratch and a NumPy implementation of Scaled Dot-Product Attention.

## Directory Structure

```text
.
├── src/
│   └── ngram_model.py       # Task 1: The Trigram Model Class
├── tests/
│   └── test_ngram.py        # Task 1: Execution/Test script
├── task_2.py                # Task 2: Attention Mechanism & Demo
├── README.md
└── evaluation.md
```

## Prerequisites

- Python 3.7+
- NumPy (Required for Task 2 only)

```bash
pip install numpy
```

## How to Run

### Task 1: Trigram Language Model

To run the Trigram model (training on sample text and generating new sentences), execute the test script from the root directory of the project. 

*Note: Running from the root is necessary to ensure Python resolves imports correctly.*

```bash
# Run as a module
python3 -m tests.test_ngram
```

Alternatively, if you have setup your PYTHONPATH:
```bash
python3 tests/test_ngram.py
```

### Task 2: Scaled Dot-Product Attention

To see the demonstration of the Attention mechanism (including standard and masked attention), run the standalone script:

```bash
python3 task_2.py
```

## Design Choices

Please refer to `evaluation.md` for a detailed summary of the engineering decisions, data structures, and mathematical implementations used in this project.

