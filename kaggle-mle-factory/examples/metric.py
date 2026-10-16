"""An ID-aligned RMSE evaluator for the synthetic regression example."""


def validate_submission(submission_df, answer_df):
    if list(submission_df.columns) != ["id", "target"]:
        raise ValueError("Expected columns: id,target")
    if submission_df.id.duplicated().any() or answer_df.id.duplicated().any():
        raise ValueError("Duplicate IDs")
    if len(submission_df) != len(answer_df) or set(submission_df.id) != set(answer_df.id):
        raise ValueError("Submission IDs do not match the answer IDs")
    values = submission_df.target.astype(float)
    if values.isna().any() or not values.abs().lt(float("inf")).all():
        raise ValueError("Predictions must be finite")
    return "Valid"


def grade(submission_df, answer_df):
    validate_submission(submission_df, answer_df)
    predictions = submission_df.set_index("id").target
    answers = answer_df.set_index("id").target
    error = predictions.reindex(answers.index) - answers
    score = float((error.pow(2).mean()) ** 0.5)
    return score, False
