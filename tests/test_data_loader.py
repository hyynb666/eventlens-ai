import pytest

from eventlens.data_loader import CSVLoadError, load_csv


def test_loads_valid_csv_bytes():
    dataframe = load_csv(b"name,value\nalpha,1\nbeta,2\n", filename="data.csv")
    assert dataframe.to_dict(orient="records") == [{"name": "alpha", "value": 1}, {"name": "beta", "value": 2}]


def test_rejects_empty_csv():
    with pytest.raises(CSVLoadError, match="empty"):
        load_csv(b"", filename="empty.csv")


def test_rejects_csv_without_data_rows():
    with pytest.raises(CSVLoadError, match="no data rows"):
        load_csv(b"name,value\n", filename="empty.csv")


def test_rejects_unsupported_file_type():
    with pytest.raises(CSVLoadError, match="Unsupported file type"):
        load_csv(b"name\nalpha\n", filename="data.xlsx")


def test_rejects_malformed_csv():
    with pytest.raises(CSVLoadError, match="parse"):
        load_csv(b"a,b\n1,2\n3,4,5\n", filename="bad.csv")
