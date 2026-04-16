from pathlib import Path
from typing import Union

import re

def parse_label(data: Union[Path, str]):
    try:
        content = data.read_text() if isinstance(data, Path) else data
        return [x for x in re.split(r"\s+", content) if x]
    except Exception as e:
        print(f"Error parsing label {data}: {e}")
        return []
