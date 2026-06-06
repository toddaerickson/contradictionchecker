from benchmarks.definition_eval.harness import _metrics


def _pred(label, predicted):
    return {"label": label, "predicted": predicted}


def test_metrics_perfect():
    preds = [_pred("divergent", "divergent"), _pred("consistent", "consistent")]
    m = _metrics(preds)
    assert m["confusion"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 1}
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_metrics_mixed():
    preds = [
        _pred("divergent", "divergent"),
        _pred("divergent", "consistent"),
        _pred("consistent", "divergent"),
        _pred("consistent", "consistent"),
    ]
    m = _metrics(preds)
    assert m["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5


def test_metrics_no_positive_predictions_is_none():
    preds = [_pred("consistent", "consistent")]
    m = _metrics(preds)
    assert m["precision"] is None
    assert m["recall"] is None
    assert m["f1"] is None


def test_metrics_all_divergent_perfect():
    preds = [_pred("divergent", "divergent"), _pred("divergent", "divergent")]
    m = _metrics(preds)
    assert m["confusion"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 0}
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_metrics_all_positives_wrong_is_zero_not_none():
    # tp=0 but predictions exist in both classes → F1 must be 0.0, not None.
    preds = [_pred("divergent", "consistent"), _pred("consistent", "divergent")]
    m = _metrics(preds)
    assert m["confusion"] == {"tp": 0, "fp": 1, "fn": 1, "tn": 0}
    assert m["precision"] == 0.0  # tp/(tp+fp) = 0/1
    assert m["recall"] == 0.0  # tp/(tp+fn) = 0/1
    assert m["f1"] == 0.0
