import io

import pytest

from core import db
from core.importer import import_sales, read_sales


def csv_bytes(text: str) -> io.BytesIO:
    return io.BytesIO(text.encode("utf-8"))


def test_csv_decimal_comma_is_not_multiplied_by_100():
    data = csv_bytes(
        'Номер студента,Сумма,Курс,Время\n'
        '101,"8 950,00",ML,01.08.2026 12:00:00\n'
    )

    df = read_sales(data, filename="base.csv")

    assert df.loc[0, "amount"] == pytest.approx(8950.0)


def test_numeric_amount_stays_numeric():
    data = csv_bytes(
        "student_id,amount,course,timestamp\n"
        "101,8950.50,ML,01.08.2026 12:00:00\n"
    )

    df = read_sales(data, filename="base.csv")

    assert df.loc[0, "amount"] == pytest.approx(8950.50)


def test_bundle_rows_become_one_order(tmp_path):
    conn = db.connect(str(tmp_path / "test.db"))
    data = csv_bytes(
        'Номер студента,Сумма,Курс,Время\n'
        '101,"4 000,00",ML,01.08.2026 12:00:00\n'
        '101,"3 500,00",Аналитика,01.08.2026 12:00:00\n'
    )

    result = import_sales(conn, data, salt="test-salt", filename="base.csv")

    assert result["rows"] == 2
    assert result["orders"] == 1
    assert result["added"] == 1
    assert result["revenue"] == pytest.approx(7500.0)

    order = conn.execute("SELECT amount FROM fact_order").fetchone()
    items = conn.execute("SELECT COUNT(*) FROM fact_order_item").fetchone()[0]

    assert order["amount"] == pytest.approx(7500.0)
    assert items == 2


def test_reimport_is_idempotent(tmp_path):
    conn = db.connect(str(tmp_path / "test.db"))
    payload = (
        'Номер студента,Сумма,Курс,Время\n'
        '101,"7 500,00",ML,01.08.2026 12:00:00\n'
    )

    first = import_sales(conn, csv_bytes(payload), salt="test-salt", filename="base.csv")
    second = import_sales(conn, csv_bytes(payload), salt="test-salt", filename="base.csv")

    assert first["added"] == 1
    assert second["added"] == 0
    assert second["skipped"] == 1
    assert conn.execute("SELECT COUNT(*) FROM fact_order").fetchone()[0] == 1
