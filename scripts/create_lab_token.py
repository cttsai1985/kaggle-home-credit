"""Create a local Jupyter login token without printing it."""

import os
import secrets
from pathlib import Path

path = Path(".local/jupyter_token")
path.parent.mkdir(exist_ok=True)
if path.exists():
    raise SystemExit("Token already exists; retained existing login")
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w") as stream:
    stream.write(secrets.token_urlsafe(32) + "\n")
print("Created .local/jupyter_token with owner-only permissions")
