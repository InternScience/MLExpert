"""Deterministic preparation for the synthetic regression example."""
import pandas as pd


def prepare(raw_dir, public_dir, private_dir):
    data = pd.read_csv(raw_dir / "train.csv")
    heldout = data.sample(frac=0.2, random_state=42)
    train = data.drop(heldout.index)
    train.to_csv(public_dir / "train.csv", index=False)
    heldout.drop(columns="target").to_csv(public_dir / "test.csv", index=False)
    answer = heldout[["id", "target"]]
    answer.to_csv(private_dir / "test_answer.csv", index=False)
    answer.assign(target=0.0).to_csv(public_dir / "sample_submission.csv", index=False)
    assert set(train.id).isdisjoint(heldout.id)
