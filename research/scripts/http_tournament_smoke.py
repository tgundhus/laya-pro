"""Measure local HTTP tournament costs with a supplied, revision-labelled checkpoint.

Run from the repository with --checkpoint PATH --checkpoint-revision SHA --output PATH.
The fixed sample questions verify transport and finalist parity, not classifier accuracy.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        parser.error("checkpoint must be a local directory")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["LAYA_TOURNAMENT_GROUP"] = "16"
    os.environ.pop("LAYA_API_KEY", None)
    os.environ.pop("LAYA_JEV_STRICT", None)
    import torch
    from fastapi.testclient import TestClient
    from laya import Agent
    from laya.serve import create_app
    from laya.shortlist import predict_tournament

    torch.set_num_threads(2)
    agent = Agent(str(args.checkpoint), device="cpu", compile=False)

    class LocalRouter:
        loaded = ["english"]

        def predict(self, state, questions, model=None, **kwargs):
            if model not in (None, "english"):
                raise ValueError("this benchmark supplies only the English checkpoint")
            return agent.predict(state, questions, **kwargs)

    labels = ["billing", "refunds", "fraud", "cards", "transfers", "loans", "savings", "kyc", "support", "legal",
              "sales", "tech", "identity", "fees", "cash", "statements", "login", "cancel", "lost", "other"]
    state = {"message": "I was charged twice for my subscription. Please refund the duplicate payment."}
    questions = {"intent": {"type": "choice", "instructions": "What does the customer need?",
                             "criteria": {label: label for label in labels}}}
    router = LocalRouter()
    expected = predict_tournament(router, state, questions, group_size=16)
    with TestClient(create_app(router)) as client:
        def run():
            response = client.post("/v1/systemone", json={"state": state, "questions": questions, "model": "english"})
            response.raise_for_status()
            return response.json()

        result = run()
        answer = result["answers"]["intent"]
        reference = expected["answers"]["intent"]
        finalists = expected["tournament"]["intent"]["labels"]
        assert answer["choice"] == reference["choice"]
        assert answer["confidence"] == reference["confidence"]
        assert answer["answer_confidence"] == reference["answer_confidence"]
        assert answer["x_tournament"]["finalists"] == finalists
        assert list(answer["probabilities"]) == labels
        assert answer["probabilities"] == {label: reference["probabilities"].get(label, 0.0) for label in labels}
        assert abs(sum(answer["probabilities"].values()) - 1) < 0.001
        # Compare the exact same 20-label request in one pass and in the HTTP tournament.
        timings = {"one_pass": [], "http_tournament": []}
        for _ in range(2):
            router.predict(state, questions)
            run()
        for repeat in range(3):
            modes = ["one_pass", "http_tournament"] if repeat % 2 == 0 else ["http_tournament", "one_pass"]
            for mode in modes:
                start = time.perf_counter()
                router.predict(state, questions) if mode == "one_pass" else run()
                timings[mode].append((time.perf_counter() - start) * 1000)
        wide = {"intent": {**questions["intent"], "criteria": {"queue-%03d" % i: "Queue %d" % i for i in range(128)}}}
        response = client.post("/v1/systemone", json={"state": state, "questions": wide, "model": "english"})
        response.raise_for_status()
        large_answer = response.json()["answers"]["intent"]
        assert len(large_answer["probabilities"]) == 128
        assert len(large_answer["x_tournament"]["finalists"]) == 8
        assert large_answer["choice"] in large_answer["x_tournament"]["finalists"]

    output = {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
              "checkpoint_revision": args.checkpoint_revision, "device": "cpu", "torch_threads": 2,
              "warmups_per_mode": 2, "repeats": 3, "group_size": 16, "choice_options": 20,
              "samples_ms": timings, "median_ms": {mode: statistics.median(samples) for mode, samples in timings.items()},
              "finalist_probability_max_difference": 0.0, "large_choice_options": 128,
              "large_finalists": 8, "limitations": "Small fixed workload on an active workstation; HTTP mode includes transport. No accuracy claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
