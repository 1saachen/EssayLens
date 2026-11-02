"""Read training configurations, including legacy key spellings."""

import json
from pathlib import Path


def normalize_config(config):
    """Use lowercase training keys; reject conflicting legacy aliases."""
    result = dict(config)
    names = (
        "model_name", "max_length", "graph_in_dim", "graph_hidden",
        "graph_out", "num_features", "dropout", "freeze_bert_layers",
        "prompt_relevance_weight",
    )
    aliases = {name: (name.upper(),) for name in names}
    aliases.update(score_min=("min_score", "MIN_SCORE"),
                   score_max=("max_score", "MAX_SCORE"))
    for canonical, alternatives in aliases.items():
        present = [key for key in (canonical, *alternatives) if key in config]
        if not present:
            continue
        value = config[present[0]]
        if any(config[key] != value for key in present[1:]):
            raise ValueError(f"Conflicting configuration keys: {', '.join(present)}")
        result[canonical] = value
        for key in alternatives:
            result.pop(key, None)
    if "model_name" not in result:
        raise ValueError("Model configuration must specify model_name (or MODEL_NAME).")
    if "score_min" in result and "score_max" in result:
        if not result["score_min"] < result["score_max"]:
            raise ValueError("score_min must be smaller than score_max.")
    return result


def read_model_config(config_path):
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing model configuration: {path.resolve()}. "
            "Restore the trained model artifacts or run grade/training/train.py "
            "with the training dataset first. See README.md."
        )
    with path.open(encoding="utf-8") as source:
        return normalize_config(json.load(source))
