"""Generate persona and no-persona responses for each model, then judge them all. Rerun to resume or fill gaps."""
import get_responses
import judge_responses

MODELS = ("claude-opus-5-5", "gpt-6.1-sol")

if __name__ == "__main__":
    for model in MODELS:
        for baseline in (False, True):
            print(f"responses: {model}{' (no persona)' if baseline else ''}")
            get_responses.main(model, baseline)
    print("judging")
    judge_responses.main(MODELS)
