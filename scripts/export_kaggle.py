"""Stage a code wheel and offline notebook locally without publishing anything."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from home_credit.pipeline import sha256, source_identity

output = Path("kaggle-bundle")
assets = output / "code-dataset"
kernel = output / "kernel"
assets.mkdir(parents=True, exist_ok=True)
kernel.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as staging:
    subprocess.run(["uv", "build", "--wheel", "--out-dir", staging], check=True)
    wheels = list(Path(staging).glob("*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]
    shutil.copy2(wheel, assets / wheel.name)
    receipt = {
        "wheel": wheel.name,
        "sha256": sha256(wheel),
        "repository": "https://github.com/cttsai1985/kaggle-home-credit",
        "source": source_identity(),
    }
(assets / "source.json").write_text(json.dumps(receipt, indent=2) + "\n")
shutil.copy2("LICENSE", assets / "LICENSE")
for name in ["baseline.json", "reference.json"]:
    shutil.copy2(Path("configs") / name, assets / name)
receipt["configuration_sha256"] = {
    name: sha256(assets / name) for name in ["baseline.json", "reference.json"]
}
(assets / "source.json").write_text(json.dumps(receipt, indent=2) + "\n")
(assets / "dataset-metadata.json").write_text(
    json.dumps(
        {
            "id": "cttsai/home-credit-risk-code",
            "title": "Home Credit risk code",
            "licenses": [{"name": "apache-2.0"}],
        },
        indent=2,
    )
    + "\n"
)
for name in ["rerun.ipynb", "kernel-metadata.json"]:
    shutil.copy2(Path("notebooks/kaggle") / name, kernel / name)
print(f"Prepared code-only assets in {output}; no Kaggle resources changed")
