from __future__ import annotations

import json
from pathlib import Path

from refua_campaign.cli import main
from refua_campaign.clinical_trials import ClawCuresClinicalController

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def test_clinical_controller_crud_and_simulation(tmp_path: Path) -> None:
    store_path = tmp_path / "clinical_store.json"
    controller = ClawCuresClinicalController(
        workspace_root=WORKSPACE_ROOT,
        store_path=store_path,
    )

    created = controller.add_trial(
        trial_id="cc-demo",
        indication="Immunology",
        phase="Phase II",
        objective="Manage trial in ClawCures",
        status="planned",
    )
    assert created["trial"]["trial_id"] == "cc-demo"

    controller.update_trial(
        "cc-demo",
        updates={
            "status": "active",
            "config": {
                "replicates": 6,
                "enrollment": {"total_n": 60},
                "adaptive": {"burn_in_n": 20, "interim_every": 20},
            },
        },
    )

    enrolled = controller.enroll_patient(
        "cc-demo",
        patient_id="human-001",
        source="human",
        arm_id="control",
        demographics={"age": 58},
    )
    assert enrolled["patient"]["patient_id"] == "human-001"

    controller.add_result(
        "cc-demo",
        patient_id="human-001",
        values={
            "arm_id": "control",
            "change": 5.0,
            "responder": False,
            "safety_event": False,
        },
    )

    simulation = controller.simulate_trial("cc-demo", replicates=3, seed=7)
    assert simulation["simulation"]["summary"]["blended_effect_estimate"] is not None

    listing = controller.list_trials()
    assert listing["count"] == 1

    removed = controller.remove_trial("cc-demo")
    assert removed["removed"] is True


def test_clinical_cli_commands(tmp_path: Path, capsys: object) -> None:
    store_path = tmp_path / "clinical_store.json"

    rc = main(
        [
            "trials-add",
            "--trial-id",
            "cli-demo",
            "--store",
            str(store_path),
        ]
    )
    assert rc == 0
    _ = capsys.readouterr()

    rc = main(
        [
            "trials-enroll",
            "--trial-id",
            "cli-demo",
            "--patient-id",
            "cli-human-1",
            "--source",
            "human",
            "--arm-id",
            "control",
            "--demographics-json",
            '{"age": 61}',
            "--store",
            str(store_path),
        ]
    )
    assert rc == 0
    _ = capsys.readouterr()

    rc = main(
        [
            "trials-result",
            "--trial-id",
            "cli-demo",
            "--patient-id",
            "cli-human-1",
            "--values-json",
            '{"arm_id":"control","change":4.8,"responder":false}',
            "--store",
            str(store_path),
        ]
    )
    assert rc == 0
    _ = capsys.readouterr()

    rc = main(["trials-list", "--store", str(store_path)])
    assert rc == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["count"] >= 1
