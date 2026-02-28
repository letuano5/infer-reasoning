# Inference Reasoning Pipeline

Multi-model Text-to-SQL inference with reasoning (Gemini 2.5 Pro, DeepSeek Reasoner).

## Setup

```bash
pip install -r requirements.txt
```

Edit `.env` and fill in your API keys:

```
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
```

Place SQLite database files in `sqlite/` directory. Each file should be named `<schema_id>.sqlite` matching the `schema_id` column in `questions.csv`.

## Run

Run all models on the first 100 questions, 4 threads per model:

```bash
python runner.py --models gemini deepseek --limit 100 --workers 4
```

Run a single model:

```bash
python runner.py --models gemini --limit 50 --workers 2
```

Adjust checkpoint frequency (save every 5 questions):

```bash
python runner.py --models deepseek --limit 200 --workers 4 --checkpoint-every 5
```

## Output

Each model produces `questions_<model>.csv` (e.g. `questions_gemini.csv`, `questions_deepseek.csv`) with columns from the original `questions.csv` plus filled `sql_answer` and `think` columns.

## Resume

Re-running the same command skips questions that already have `sql_answer` in the output CSV. Delete the output file to start fresh.

## Arguments

| Arg | Required | Default | Description |
|-----|----------|---------|-------------|
| `--models` | No | `gemini deepseek` | Models to run |
| `--limit` | Yes | -- | Max questions to process |
| `--workers` | No | `4` | Threads per model |
| `--checkpoint-every` | No | `10` | Save interval |
