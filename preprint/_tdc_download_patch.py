"""Work around the Harvard Dataverse WAF that 403-blocks python-requests.

TDC (PyTDC 1.1.x) downloads its datasets from
``https://dataverse.harvard.edu/api/access/datafile/<id>`` using the default
``python-requests`` User-Agent. Dataverse's WAF forbids that agent and returns a
199-byte "403 Forbidden" HTML page, which TDC silently writes as the ``.tab``
file (later parsed as a one-line HTML DOCTYPE, crashing the adapter). ``curl``
and browsers succeed because they send a normal User-Agent.

Importing this module patches ``requests.get`` in-process to inject a browser
User-Agent header (redirects to the S3 pre-signed URL are already followed by
requests' defaults). Import it BEFORE constructing any TDC-backed adapter.
"""

from __future__ import annotations

import requests

_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_PATCH_FLAG = "_snptx_ua_patched"


def apply() -> None:
    """Idempotently patch ``requests.get`` to send a browser User-Agent."""
    if getattr(requests.get, _PATCH_FLAG, False):
        return

    _orig_get = requests.get

    def _get_with_ua(url, **kwargs):  # noqa: ANN001, ANN202
        headers = dict(kwargs.pop("headers", None) or {})
        headers.setdefault("User-Agent", _BROWSER_UA)
        return _orig_get(url, headers=headers, **kwargs)

    setattr(_get_with_ua, _PATCH_FLAG, True)
    requests.get = _get_with_ua  # type: ignore[assignment]


apply()
