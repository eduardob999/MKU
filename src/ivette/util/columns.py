"""Training-target column detection shared by the model/SDF pipelines."""

# Tokens whose presence marks a column as a bioactivity target (ChEMBL assays,
# IC50/EC50/Ki/Kd/Potency endpoints, etc.).
TARGET_TOKENS = ("ChEMBL:", "IC50", "EC50", "Ki", "Kd", "Potency")


def is_target_column(col):
    """True if ``col`` looks like a bioactivity target column."""
    return any(token in col for token in TARGET_TOKENS)


def select_targets(df, min_coverage):
    """Target columns in ``df`` whose non-null fraction meets ``min_coverage``."""
    return [
        col for col in df.columns
        if is_target_column(col) and df[col].notna().mean() >= min_coverage
    ]
