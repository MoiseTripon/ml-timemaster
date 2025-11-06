from typing import Any, Dict, List, Optional

class Output:
    DICT: int
    STRING: int
    BYTES: int
    DATAFRAME: int

def image_to_data(
    image: Any,
    *,
    lang: Optional[str] = ...,
    config: Optional[str] = ...,
    output_type: int = ...,
) -> Dict[str, List[str]]: ...

def image_to_string(
    image: Any,
    *,
    lang: Optional[str] = ...,
    config: Optional[str] = ...,
) -> str: ...