import pandas as pd
import pytest

from core.attribution import compute


def touches(rows):
    return pd.DataFrame(
        rows,
        columns=["touch_id", "user_id_hash", "placement_id", "timestamp"],
    ).assign(timestamp=lambda x: pd.to_datetime(x["timestamp"]))


def orders(rows):
    return pd.DataFrame(
        rows,
        columns=["order_id", "user_id_hash", "amount", "variable_costs", "timestamp"],
    ).assign(timestamp=lambda x: pd.to_datetime(x["timestamp"]))


def test_last_touch_gives_all_credit_to_latest_touch():
    t = touches([
        (1, "u1", "A", "2026-08-01"),
        (2, "u1", "B", "2026-08-05"),
    ])
    o = orders([
        ("o1", "u1", 10000.0, 0.0, "2026-08-10"),
    ])

    result = compute(t, o, model="last", window_days=30)

    assert len(result) == 1
    assert result.iloc[0]["placement_id"] == "B"
    assert result.iloc[0]["weight"] == pytest.approx(1.0)
    assert result.iloc[0]["attributed_revenue"] == pytest.approx(10000.0)


def test_first_touch_gives_all_credit_to_first_touch():
    t = touches([
        (1, "u1", "A", "2026-08-01"),
        (2, "u1", "B", "2026-08-05"),
    ])
    o = orders([
        ("o1", "u1", 10000.0, 0.0, "2026-08-10"),
    ])

    result = compute(t, o, model="first", window_days=30)

    assert len(result) == 1
    assert result.iloc[0]["placement_id"] == "A"
    assert result.iloc[0]["weight"] == pytest.approx(1.0)


def test_linear_splits_credit_equally():
    t = touches([
        (1, "u1", "A", "2026-08-01"),
        (2, "u1", "B", "2026-08-05"),
    ])
    o = orders([
        ("o1", "u1", 10000.0, 1000.0, "2026-08-10"),
    ])

    result = compute(t, o, model="linear", window_days=30).sort_values("placement_id")

    assert result["weight"].tolist() == pytest.approx([0.5, 0.5])
    assert result["attributed_revenue"].tolist() == pytest.approx([5000.0, 5000.0])
    assert result["attributed_margin"].tolist() == pytest.approx([4500.0, 4500.0])


def test_position_model_uses_40_20_40_for_three_touches():
    t = touches([
        (1, "u1", "A", "2026-08-01"),
        (2, "u1", "B", "2026-08-03"),
        (3, "u1", "C", "2026-08-05"),
    ])
    o = orders([
        ("o1", "u1", 10000.0, 0.0, "2026-08-10"),
    ])

    result = compute(t, o, model="position", window_days=30).sort_values("placement_id")

    assert result["weight"].tolist() == pytest.approx([0.4, 0.2, 0.4])
    assert result["weight"].sum() == pytest.approx(1.0)


def test_touch_older_than_window_becomes_organic():
    t = touches([
        (1, "u1", "A", "2026-07-01"),
    ])
    o = orders([
        ("o1", "u1", 10000.0, 0.0, "2026-08-10"),
    ])

    result = compute(t, o, model="last", window_days=30)

    assert len(result) == 1
    assert pd.isna(result.iloc[0]["placement_id"])
    assert result.iloc[0]["weight"] == pytest.approx(1.0)


def test_repeat_purchase_does_not_reuse_old_touch():
    t = touches([
        (1, "u1", "A", "2026-08-01"),
    ])
    o = orders([
        ("o1", "u1", 10000.0, 0.0, "2026-08-05"),
        ("o2", "u1", 7000.0, 0.0, "2026-08-20"),
    ])

    result = compute(t, o, model="last", window_days=30)

    first = result[result["order_id"] == "o1"].iloc[0]
    second = result[result["order_id"] == "o2"].iloc[0]

    assert first["placement_id"] == "A"
    assert pd.isna(second["placement_id"])


def test_touch_after_first_purchase_can_credit_second_purchase():
    t = touches([
        (1, "u1", "A", "2026-08-01"),
        (2, "u1", "B", "2026-08-10"),
    ])
    o = orders([
        ("o1", "u1", 10000.0, 0.0, "2026-08-05"),
        ("o2", "u1", 7000.0, 0.0, "2026-08-20"),
    ])

    result = compute(t, o, model="last", window_days=30)

    first = result[result["order_id"] == "o1"].iloc[0]
    second = result[result["order_id"] == "o2"].iloc[0]

    assert first["placement_id"] == "A"
    assert second["placement_id"] == "B"


def test_unknown_model_raises_error():
    t = touches([(1, "u1", "A", "2026-08-01")])
    o = orders([("o1", "u1", 10000.0, 0.0, "2026-08-10")])

    with pytest.raises(ValueError):
        compute(t, o, model="unknown", window_days=30)
