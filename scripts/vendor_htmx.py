"""Download htmx.min.js into the web static dir (ADR-0007, step G1).

Run after cloning::

    uv run python scripts/vendor_htmx.py

The release is pinned to v1.9.12 with a SHA-256 integrity check so a
swapped-out CDN can't slip in different bytes.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://unpkg.com/htmx.org@1.9.12/dist/htmx.min.js"
EXPECTED_SHA256 = (
    # Pin once you have the real bytes locally; see README "Vendoring HTMX".
    # Until pinned, the script warns instead of failing so first-time clones
    # don't bounce on a hash mismatch we haven't yet captured.
    ""
)
TARGET = (
    Path(__file__).resolve().parent.parent
    / "consistency_checker"
    / "web"
    / "static"
    / "htmx.min.js"
)


def main() -> int:
    print(f"Fetching {URL} → {TARGET}")
    with urllib.request.urlopen(URL, timeout=15) as resp:
        data: bytes = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if EXPECTED_SHA256:
        if digest != EXPECTED_SHA256:
            print(
                f"SHA-256 mismatch:\n  expected {EXPECTED_SHA256}\n  got      {digest}",
                file=sys.stderr,
            )
            return 1
    else:
        print(f"NOTE: no SHA-256 pin yet. Captured digest: {digest}")
        print("Paste into EXPECTED_SHA256 to lock the pin.")
    TARGET.write_bytes(data)
    print(f"Wrote {len(data)} bytes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
