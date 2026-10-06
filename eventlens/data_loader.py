"""CSV loading helpers used by the application and other Python callers."""

from __future__ import annotations

from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import BinaryIO

import pandas as pd
from pandas.errors import EmptyDataError, ParserError


class CSVLoadError(ValueError):
    """Raised when a CSV cannot be loaded as a usable dataset."""


def load_csv(source: str | PathLike[str] | bytes | bytearray | BinaryIO, filename: str | None = None) -> pd.DataFrame:
    """Load a CSV from a path, bytes, or binary file-like object.

    UTF-8 (including a BOM) is tried first, followed by Windows-1252 for
    common files exported from Windows applications.
    """
    source_name = filename or (str(source) if isinstance(source, (str, PathLike)) else "uploaded file")
    if filename and Path(filename).suffix.lower() != ".csv":
        raise CSVLoadError("Unsupported file type. Please upload a .csv file.")
    if not filename and isinstance(source, (str, PathLike)) and Path(source).suffix and Path(source).suffix.lower() != ".csv":
        raise CSVLoadError("Unsupported file type. Please choose a file with the .csv extension.")

    try:
        if isinstance(source, (str, PathLike)):
            raw = Path(source).read_bytes()
        elif isinstance(source, (bytes, bytearray)):
            raw = bytes(source)
        elif hasattr(source, "getvalue"):
            raw = source.getvalue()
        else:
            raw = source.read()
    except OSError as exc:
        raise CSVLoadError(f"Could not read {source_name}: {exc}") from exc

    if not raw:
        raise CSVLoadError("The CSV file is empty.")

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        try:
            dataframe = pd.read_csv(BytesIO(text.encode("utf-8")))
        except EmptyDataError as exc:
            raise CSVLoadError("The CSV file does not contain a header row or data.") from exc
        except ParserError as exc:
            raise CSVLoadError(f"Could not parse the CSV file. Check that each row has the same number of fields. Details: {exc}") from exc
        except (UnicodeError, ValueError) as exc:
            last_error = exc
            continue
        if dataframe.empty:
            raise CSVLoadError("The CSV file has column headers but contains no data rows.")
        return dataframe

    raise CSVLoadError(f"Could not decode the CSV file. Save it as UTF-8 or Windows-1252 and try again. Details: {last_error}")
