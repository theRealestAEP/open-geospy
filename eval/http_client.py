import json
import mimetypes
import os
import uuid
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple


def _encode_multipart(
    *,
    fields: Dict[str, str],
    file_field: str,
    file_path: str,
) -> Tuple[bytes, str]:
    boundary = f"geospy-{uuid.uuid4().hex}"
    mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    filename = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        payload = f.read()

    chunks: List[bytes] = []
    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
            payload,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks), boundary


def post_image_for_json(
    *,
    url: str,
    file_path: str,
    fields: Dict[str, str],
    timeout_seconds: float,
) -> Tuple[int, dict]:
    body, boundary = _encode_multipart(
        fields=fields,
        file_field="image",
        file_path=file_path,
    )
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
            status = int(response.status)
            payload = json.loads(response.read().decode("utf-8"))
            return status, payload
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"detail": raw}
        return int(exc.code), payload


def maybe_str(value: Optional[object]) -> str:
    if value is None:
        return ""
    return str(value)
