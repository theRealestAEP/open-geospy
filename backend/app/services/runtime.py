import os
from typing import Optional

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def parse_boolish(value: Optional[object], default: bool = False) -> bool:
    if value is None:
        return bool(default)
    raw = str(value).strip().lower()
    if raw in TRUE_VALUES:
        return True
    if raw in FALSE_VALUES:
        return False
    return bool(default)


def env_bool(name: str, default: bool = False) -> bool:
    return parse_boolish(os.getenv(name), default=default)


def env_int(name: str, default: int, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    value = int(os.getenv(name, str(default)))
    if minimum is not None:
        value = max(int(minimum), value)
    if maximum is not None:
        value = min(int(maximum), value)
    return value


def env_float(
    name: str,
    default: float,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    value = float(os.getenv(name, str(default)))
    if minimum is not None:
        value = max(float(minimum), value)
    if maximum is not None:
        value = min(float(maximum), value)
    return value


def resolve_torch_device(torch, requested: str = "auto") -> str:
    raw = str(requested or "auto").strip().lower()
    if raw and raw != "auto":
        return raw
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
