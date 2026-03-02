from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Any


def build_regulatory_bundle(
    *,
    payload: dict[str, Any],
    bundle_dir: Path,
    campaign_run_path: Path | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    bundle_api = _resolve_regulatory_api()
    build_evidence_bundle = bundle_api["build_evidence_bundle"]
    verify_evidence_bundle = bundle_api["verify_evidence_bundle"]

    input_path = campaign_run_path
    temp_path: Path | None = None
    if input_path is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="clawcures_reg_bundle_"))
        temp_path = temp_dir / "campaign_run.json"
        temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        input_path = temp_path

    manifest = build_evidence_bundle(
        campaign_run_path=input_path,
        output_dir=bundle_dir,
        include_checklists=True,
        checklist_templates=["drug_discovery_comprehensive"],
        overwrite=bool(overwrite),
    )
    verification = verify_evidence_bundle(bundle_dir)

    result = {
        "bundle_dir": str(bundle_dir.expanduser().resolve()),
        "manifest": manifest,
        "verification": {
            "ok": bool(getattr(verification, "ok", False)),
            "checked_files": int(getattr(verification, "checked_files", 0)),
            "errors": list(getattr(verification, "errors", ()) or ()),
            "warnings": list(getattr(verification, "warnings", ()) or ()),
        },
    }
    if temp_path is not None:
        result["temporary_campaign_run_path"] = str(temp_path)
    return result


def _resolve_regulatory_api() -> dict[str, Any]:
    try:
        from refua_regulatory.bundle import (  # type: ignore
            build_evidence_bundle,
            verify_evidence_bundle,
        )

        return {
            "build_evidence_bundle": build_evidence_bundle,
            "verify_evidence_bundle": verify_evidence_bundle,
        }
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[3]
        local_src = repo_root / "refua-regulatory" / "src"
        if local_src.exists():
            sys.path.insert(0, str(local_src))
            from refua_regulatory.bundle import (  # type: ignore
                build_evidence_bundle,
                verify_evidence_bundle,
            )

            return {
                "build_evidence_bundle": build_evidence_bundle,
                "verify_evidence_bundle": verify_evidence_bundle,
            }
        raise RuntimeError(
            "refua-regulatory is not available. Install it to enable bundle generation."
        )
