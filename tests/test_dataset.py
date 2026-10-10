import pytest

from smart_ppe.data.validate_dataset import (prepare_dataset, read_class_names, validate_dataset,
                                             validate_label_file)
from smart_ppe.utils.paths import apply_overrides, resolve_dataset_root


def test_valid_dataset_passes_and_counts(dataset, ds_cfg):
    rep = validate_dataset(dataset, ds_cfg)
    assert rep["errors"] == []
    assert {s: v["images"] for s, v in rep["splits"].items()} == {"train": 6, "val": 3, "test": 3}
    assert set(rep["classes"].values()) == {"helmet", "head", "person"}


@pytest.mark.parametrize("line,fragment", [
    ("5 0.5 0.5 0.1 0.1", "class id 5"),
    ("0 0.5 0.5 0.0 0.1", "non-positive"),
    ("0 1.5 0.5 0.1 0.1", "center outside"),
    ("0 0.98 0.5 0.1 0.1", "extends outside"),
    ("0 0.5 0.5 0.1", "expected 5 fields"),
    ("a 0.5 0.5 0.1 0.1", "non-numeric"),
])
def test_label_errors_detected(tmp_path, line, fragment):
    p = tmp_path / "x.txt"
    p.write_text(line + "\n")
    errors, counts = validate_label_file(p, num_classes=3)
    assert len(errors) == 1 and fragment in errors[0]
    assert sum(counts.values()) == 0


def test_valid_label_counts_classes(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("0 0.5 0.5 0.2 0.2\n2 0.3 0.3 0.1 0.1\n")
    errors, counts = validate_label_file(p, 3)
    assert errors == [] and counts[0] == 1 and counts[2] == 1


def test_orphan_label_and_missing_split(dataset, ds_cfg):
    (dataset / "test" / "images" / "test_0000.jpg").unlink()
    assert any("without image" in e for e in validate_dataset(dataset, ds_cfg)["errors"])
    import shutil
    shutil.rmtree(dataset / "test")
    assert any("split 'test'" in e for e in validate_dataset(dataset, ds_cfg)["errors"])


def test_leakage_detected(dataset, ds_cfg):
    import shutil
    shutil.copy(dataset / "train" / "images" / "train_0000.jpg", dataset / "test" / "images" / "dup.jpg")
    assert any("leakage" in e for e in validate_dataset(dataset, ds_cfg)["errors"])


def test_missing_role_class(dataset, ds_cfg):
    ds_cfg["roles"]["vest"] = "vest"
    assert any("role 'vest'" in e for e in validate_dataset(dataset, ds_cfg)["errors"])


def test_class_names_list_and_dict(tmp_path):
    (tmp_path / "a.yaml").write_text("names: [a, b]\n")
    (tmp_path / "b.yaml").write_text("names: {0: a, 1: b}\n")
    assert read_class_names(tmp_path / "a.yaml") == read_class_names(tmp_path / "b.yaml") == {0: "a", 1: "b"}


def test_dataset_root_from_env_and_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("PPE_DATASET_ROOT", raising=False)
    with pytest.raises(ValueError, match="PPE_DATASET_ROOT"):
        resolve_dataset_root({"root": None})
    with pytest.raises(FileNotFoundError):
        resolve_dataset_root({"root": str(tmp_path / "nope")})
    monkeypatch.setenv("PPE_DATASET_ROOT", str(tmp_path))
    assert resolve_dataset_root({"root": "ignored"}) == tmp_path


def test_prepare_writes_resolved_yaml(dataset, ds_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("PPE_DATASET_ROOT", str(dataset))
    ds_cfg["output_dir"] = str(tmp_path / "out")
    resolved, _ = prepare_dataset(ds_cfg)
    import yaml
    d = yaml.safe_load(resolved.read_text())
    assert d["names"][2] == "person" and d["val"].endswith("images") and "valid" in d["val"]


def test_prepare_raises_on_errors(dataset, ds_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("PPE_DATASET_ROOT", str(dataset))
    ds_cfg["output_dir"] = str(tmp_path / "out")
    ds_cfg["min_images"]["train"] = 999
    with pytest.raises(ValueError, match="validation failed"):
        prepare_dataset(ds_cfg)


def test_overrides():
    cfg = apply_overrides({"a": {"b": 1}}, ["a.b=2", "c.d=[1,2]"])
    assert cfg == {"a": {"b": 2}, "c": {"d": [1, 2]}}
    with pytest.raises(ValueError):
        apply_overrides({}, ["novalue"])


def test_make_val_split_disjoint_deterministic_and_test_untouched(dataset, tmp_path):
    import shutil
    from smart_ppe.data.make_val_split import make_val_split
    shutil.rmtree(dataset / "valid")                      # simulate a train+test-only export
    a = make_val_split(dataset, tmp_path / "a", val_frac=0.5, seed=3)
    b = make_val_split(dataset, tmp_path / "b", val_frac=0.5, seed=3)
    assert a == b == {"valid": 3, "train": 3, "test": 3}
    names = lambda root, s: {p.name for p in (root / s / "images").iterdir()}
    assert names(tmp_path / "a", "valid") == names(tmp_path / "b", "valid")               # same seed, same split
    assert not names(tmp_path / "a", "valid") & names(tmp_path / "a", "train")            # disjoint
    assert names(tmp_path / "a", "test") == names(dataset, "test")                        # test untouched
    assert len(list((tmp_path / "a" / "valid" / "labels").iterdir())) == 3                # labels follow images
    with pytest.raises(ValueError):
        make_val_split(dataset, dataset)
    with pytest.raises(ValueError):
        make_val_split(dataset, tmp_path / "c", val_frac=1.5)
