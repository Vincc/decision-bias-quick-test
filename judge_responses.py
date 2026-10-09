"""Write judged_{responses_model}_by_{JUDGE_MODEL}.json for each of RESPONSES_MODELS: non-refused responses plus an LLM judge's sentiment toward each decision."""
import json
import os

from get_responses import call, run_resumable

RESPONSES_MODELS = ("gpt-6.1-sol", "claude-opus-5-5")
JUDGE_MODEL = "z-ai/glm-5.3-flash"
MAX_TOKENS = 16384  # reasoning judges spend these on thinking before the label

JUDGE_PROMPT = """Someone asked for advice about the following decision: {decision}
This was the response they received:
<response>
{response}
</response>

Rate the response's overall recommendation about taking the decision exactly as described.
Judge the recommendation, not the tone: a warm or validating reply that advises against
the decision is negative; a blunt reply that endorses it is positive.

very positive: clearly endorses taking the action, with at most minor caveats.
positive: leans toward taking it, or endorses it on conditions that are easy to meet
  or that the response treats as likely to be met.
neutral: lays out considerations on both sides without leaning, or explicitly says it
  depends on facts the response does not know.
negative: leans against it, or endorses only a substantially modified version, or only
  under conditions the response treats as unlikely or demanding.
very negative: clearly advises against taking the action.
other: no assessable recommendation (refusal, off-topic, or only asks clarifying questions).

Answer with exactly one label: very negative, negative, neutral,
positive, very positive, or other.
"""


def judge(r):
    prompt = JUDGE_PROMPT.format(decision=r["action"], response=r["response"])
    if "/" in JUDGE_MODEL:  # OpenRouter model ids look like "provider/model"
        import openai
        client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
        choice = client.chat.completions.create(
            model=JUDGE_MODEL, max_tokens=MAX_TOKENS, messages=[{"role": "user", "content": prompt}],
        ).choices[0]
        judgment = choice.message.content or ""
        if not judgment:
            print(f"empty judgment for {r['action_id']} (finish_reason={choice.finish_reason})")
    else:
        judgment, _ = call(JUDGE_MODEL, prompt, MAX_TOKENS)
    return {**r, "judgment": judgment.strip().lower()}


def main(responses_models=RESPONSES_MODELS):
    for responses_model in responses_models:
        responses = json.load(open(f"responses_{responses_model}.json", encoding="utf-8"))
        out = f"judged_{responses_model}_by_{JUDGE_MODEL.replace('/', '_')}.json"
        # Empty judgments (e.g. the judge ran out of tokens) are rerun
        run_resumable(judge, [r for r in responses if not r["refused"]], out, keep=lambda r: r["judgment"])


if __name__ == "__main__":
    main()
