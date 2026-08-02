"""Tests for the CMake model staging/downloading helper without network access."""

from __future__ import annotations

import hashlib
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "cmake" / "stage_model.cmake"


def _run_stage(
    destination: Path,
    expected_sha256: str,
    *,
    model_source: Path | None = None,
    curl_executable: Path | None = None,
    wget_executable: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "cmake",
            "-DMODEL_URL=https://invalid.example/yolo26m-seg.pt",
            f"-DMODEL_SOURCE={model_source or ''}",
            f"-DMODEL_PATH={destination}",
            f"-DMODEL_SHA256={expected_sha256}",
            f"-DCURL_EXECUTABLE={curl_executable or ''}",
            f"-DWGET_EXECUTABLE={wget_executable or ''}",
            "-P",
            str(SCRIPT),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _make_fake_curl(path: Path, source: Path) -> Path:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "output=''\n"
        "while (($#)); do\n"
        "  if [[ $1 == --output ]]; then output=$2; shift 2; else shift; fi\n"
        "done\n"
        "test -n \"$output\"\n"
        f"cp {shlex.quote(str(source))} \"$output\"\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | 0o111)
    return path


def _make_fake_wget(path: Path, source: Path) -> Path:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "output=''\n"
        "for argument in \"$@\"; do\n"
        "  case $argument in\n"
        "    --output-document=*) output=${argument#--output-document=} ;;\n"
        "  esac\n"
        "done\n"
        "test -n \"$output\"\n"
        f"cp {shlex.quote(str(source))} \"$output\"\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | 0o111)
    return path


def test_stage_script_reuses_a_valid_cached_model(tmp_path: Path) -> None:
    destination = tmp_path / "models" / "yolo26m-seg.pt"
    destination.parent.mkdir(parents=True)
    payload = b"valid cached test model\n"
    destination.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    result = _run_stage(destination, expected, curl_executable=Path("/bin/false"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert destination.read_bytes() == payload
    assert "Reusing verified model" in result.stdout + result.stderr


def test_stage_script_copies_and_verifies_an_offline_model(tmp_path: Path) -> None:
    source = tmp_path / "offline-model.pt"
    payload = b"offline model payload\n"
    source.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "models" / "yolo26m-seg.pt"

    result = _run_stage(destination, expected, model_source=source)

    assert result.returncode == 0, result.stdout + result.stderr
    assert destination.read_bytes() == payload
    assert "Staged verified model with local file" in result.stdout + result.stderr


def test_stage_script_redownloads_a_corrupt_cached_model_with_curl(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    good_payload = b"replacement model payload\n"
    source.write_bytes(good_payload)
    expected = hashlib.sha256(good_payload).hexdigest()

    destination = tmp_path / "models" / "yolo26m-seg.pt"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"corrupt partial model\n")
    fake_curl = _make_fake_curl(tmp_path / "fake-curl", source)

    result = _run_stage(destination, expected, curl_executable=fake_curl)

    assert result.returncode == 0, result.stdout + result.stderr
    assert destination.read_bytes() == good_payload
    output = result.stdout + result.stderr
    assert "Removing model with unexpected SHA-256" in output
    assert "Staged verified model with curl" in output
    assert not Path(f"{destination}.part").exists()


def test_stage_script_falls_back_to_wget(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    payload = b"wget model payload\n"
    source.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "models" / "yolo26m-seg.pt"
    fake_wget = _make_fake_wget(tmp_path / "fake-wget", source)

    result = _run_stage(destination, expected, wget_executable=fake_wget)

    assert result.returncode == 0, result.stdout + result.stderr
    assert destination.read_bytes() == payload
    assert "Staged verified model with wget" in result.stdout + result.stderr


def test_stage_script_rejects_a_checksum_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"wrong payload\n")
    expected = hashlib.sha256(b"expected payload\n").hexdigest()
    destination = tmp_path / "models" / "yolo26m-seg.pt"
    fake_curl = _make_fake_curl(tmp_path / "fake-curl", source)

    result = _run_stage(destination, expected, curl_executable=fake_curl)

    assert result.returncode != 0
    assert "failed SHA-256 verification" in result.stdout + result.stderr
    assert not destination.exists()
    assert not Path(f"{destination}.part").exists()
