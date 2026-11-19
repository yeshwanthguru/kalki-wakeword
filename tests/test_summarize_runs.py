import json

import pytest

import summarize_runs as sr


def report(auc, fa, caught=0.97, recall=0.94, fpr=0.02, eval_set="v2"):
    row = {"auc": auc, "recall@0.5": recall, "fpr@0.5": fpr, "stream": {"0.8": {"detected": caught, "false_alarms_per_hour": fa}}}
    return {"fp32": row, "int8": row, "eval_set": eval_set}


def write(tmp_path, name, **kw):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(report(**kw)))
    return str(p)


def test_groups_mean_range_and_flags_only_clear_differences(tmp_path):
    ctl = [write(tmp_path, f"c{i}", auc=a, fa=f) for i, (a, f) in enumerate([(0.9925, 47), (0.9937, 56), (0.9931, 52)])]
    same = [write(tmp_path, "s", auc=0.9933, fa=50)]
    worse = [write(tmp_path, "w", auc=0.9931, fa=80)]
    groups, sets = sr.load_groups([f"control={p}" for p in ctl] + [f"same={same[0]}", f"worse={worse[0]}"])
    assert sets == {"v2"} and len(groups["control"]) == 3
    md = sr.summarize(groups, reference="control")
    rows = {line.split("|")[1].strip(): line for line in md.splitlines() if line.startswith("| ") and "---" not in line}
    assert "0.9931 [0.9925-0.9937]" in rows["control"] and "[47-56]" in rows["control"]
    assert "*" not in rows["same"]                      # inside the control range: not a clear difference
    assert "80 *" in rows["worse"] and "AUC" not in md.split("Flagged:")[-1]   # false alarms flagged, AUC not
    assert "worse: false alarms/h @0.8" in md


def test_single_reference_run_warns_and_main_flags_mixed_sets(tmp_path, capsys):
    a = write(tmp_path, "a", auc=0.99, fa=10)
    b = write(tmp_path, "b", auc=0.99, fa=10, eval_set="v1")
    assert sr.main([f"x={a}", f"y={b}"]) == 1
    assert "different evaluation sets" in capsys.readouterr().err
    assert sr.main([f"x={a}", f"y={a}", "--reference", "x"]) == 0
    assert "Only 1 reference run" in capsys.readouterr().out


def test_bad_arguments_exit(tmp_path):
    with pytest.raises(SystemExit):
        sr.load_groups(["no-equals-sign"])
    with pytest.raises(SystemExit):
        sr.main([f"x={write(tmp_path, 'a', auc=1, fa=1)}", "--reference", "nope"])
