"""Regression checks that run without model downloads or deep learning packages."""

import ast
import contextlib
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grade.model_config import normalize_config, read_model_config


def definitions(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return ast.Module(body=[node for node in tree.body
                           if isinstance(node, ast.FunctionDef) and node.name in names],
                      type_ignores=[])


class TargetNormalizer:
    """Pickleable stand-in; the regression checks export identity, not sklearn."""
    pass


class ModelArtifactTests(unittest.TestCase):
    def test_training_configuration_reaches_inference_globals(self):
        config = dict(model_name="microsoft/deberta-v3-base", max_length=384,
                      graph_in_dim=3, graph_hidden=32, graph_out=128,
                      num_features=20, dropout=0.2, freeze_bert_layers=4,
                      prompt_relevance_weight=0.1, score_min=2, score_max=8)
        namespace = {"read_model_config": read_model_config}
        exec(compile(definitions(ROOT / "grade/grade_run.py", {"load_config"}),
                     "grade_run.py", "exec"), namespace)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(namespace["load_config"](path), config)
        for name in config:
            global_name = {"score_min": "MIN_SCORE", "score_max": "MAX_SCORE"}.get(name, name.upper())
            self.assertEqual(namespace[global_name], config[name])

    def test_legacy_configuration(self):
        result = normalize_config(dict(MODEL_NAME="roberta-base", GRAPH_OUT=64,
                                       min_score=1, max_score=5))
        self.assertEqual(result, dict(model_name="roberta-base", graph_out=64,
                                      score_min=1, score_max=5))

    def test_conflicting_aliases_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            normalize_config(dict(model_name="roberta-base", MODEL_NAME="different"))

    def test_missing_model_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "model_name"):
            normalize_config({})

    def test_invalid_score_range_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "score_min"):
            normalize_config(dict(model_name="roberta-base", score_min=5, score_max=1))

    def test_missing_artifacts_have_recovery_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "training/train.py"):
                read_model_config(Path(tmp) / "missing.json")

    def test_target_export_survives_cpu_and_amp_setup(self):
        tree = ast.parse((ROOT / "grade/training/train.py").read_text(encoding="utf-8"))
        target_init = next(node for node in tree.body if isinstance(node, ast.Assign)
                           and isinstance(node.value, ast.Call)
                           and isinstance(node.value.func, ast.Name)
                           and node.value.func.id == "MinMaxScaler")
        amp_setup = next(node for node in tree.body if isinstance(node, ast.If)
                         and isinstance(node.test, ast.Name) and node.test.id == "use_amp")
        export = next(node for node in tree.body if isinstance(node, ast.Try)
                      and any(isinstance(child, ast.Name) and child.id == "target_scaler_path"
                              for child in ast.walk(node)))
        for enabled in (False, True):
            with self.subTest(amp=enabled), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "target_scaler.pkl"
                namespace = dict(MinMaxScaler=TargetNormalizer, pickle=pickle,
                                 use_amp=enabled, target_scaler_path=path,
                                 torch=SimpleNamespace(cuda=SimpleNamespace(
                                     amp=SimpleNamespace(GradScaler=object))))
                module = ast.Module(body=[target_init, amp_setup, export], type_ignores=[])
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(module, "train.py", "exec"), namespace)
                with path.open("rb") as source:
                    self.assertIsInstance(pickle.load(source), TargetNormalizer)

    def test_full_checkpoint_exports_target_normalizer(self):
        tree = ast.parse((ROOT / "grade/training/train.py").read_text(encoding="utf-8"))
        checkpoint = next(node for node in ast.walk(tree) if isinstance(node, ast.Dict)
                          and any(isinstance(key, ast.Constant) and key.value == "model_state_dict"
                                  for key in node.keys))
        target = TargetNormalizer()
        namespace = dict(model=SimpleNamespace(state_dict=lambda: {}), model_config={},
                         feature_scaler=object(), target_scaler=target)
        value = eval(compile(ast.Expression(checkpoint), "train.py", "eval"), namespace)
        self.assertIs(value["target_scaler"], target)

    def test_training_fusion_matches_graph_output(self):
        tree = ast.parse((ROOT / "grade/training/train.py").read_text(encoding="utf-8"))
        graph = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name == "GraphEncoder")
        conv = next(node for node in ast.walk(graph) if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Attribute) and t.attr == "conv3" for t in node.targets))
        fusion = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == "FusionGraphBertAES")
        init = next(node for node in fusion.body if isinstance(node, ast.FunctionDef)
                    and node.name == "__init__")
        dimension = next(node for node in ast.walk(init) if isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Attribute) and t.attr == "fusion_input_dim"
                                 for t in node.targets))
        graph_dim = eval(compile(ast.Expression(conv.value.args[1]), "train.py", "eval"),
                         {"out_dim": 256})
        fusion_dim = eval(compile(ast.Expression(dimension.value), "train.py", "eval"),
                          dict(bert_hidden=768, graph_out=256, num_features=20))
        self.assertEqual(fusion_dim, 768 + graph_dim + 20)


if __name__ == "__main__":
    unittest.main()
