# ControlBias

Do models favor AI control measures in their advice? Each model answers decision prompts under different personas, an LLM judge rates each answer on a 5-point scale, and `analysis.ipynb` plots the results.

Needs `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `OPENROUTER_API_KEY` (for the judge).

```
uv sync
uv run python run_pipeline.py   # writes responses_*.json and judged_*.json; rerun to resume
```

Then run `analysis.ipynb`.
