from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import orjson

_BASE_OPTIONS = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY


def load_json(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def dump_json(path: Union[str, Path], obj: Any, *, indent: bool = False) -> None:
    option = _BASE_OPTIONS | (orjson.OPT_INDENT_2 if indent else 0)
    data = orjson.dumps(obj, option=option)
    with open(path, "wb") as f:
        f.write(data)


def dumps_json(obj: Any, *, indent: bool = False) -> str:
    option = _BASE_OPTIONS | (orjson.OPT_INDENT_2 if indent else 0)
    return orjson.dumps(obj, option=option).decode("utf-8")
