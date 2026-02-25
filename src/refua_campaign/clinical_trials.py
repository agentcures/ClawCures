from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


def _workspace_root_from_file() -> Path:
    # ClawCures/src/refua_campaign -> ClawCures -> refua-project
    return Path(__file__).resolve().parents[3]


class ClawCuresClinicalController:
    """Manipulate refua-clinical managed trials from ClawCures."""

    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        store_path: Path | None = None,
    ) -> None:
        self._workspace_root = workspace_root.resolve() if workspace_root else _workspace_root_from_file()
        self._store_path = store_path
        self._paths_ready = False

    def list_trials(self) -> dict[str, Any]:
        manager = self._manager()
        trials = manager.list_trials()
        return {
            "store_path": str(manager.store_path),
            "count": len(trials),
            "trials": trials,
        }

    def get_trial(self, trial_id: str) -> dict[str, Any]:
        manager = self._manager()
        trial = manager.get_trial(trial_id)
        if trial is None:
            raise KeyError(trial_id)
        return {
            "store_path": str(manager.store_path),
            "trial": trial,
        }

    def add_trial(
        self,
        *,
        trial_id: str | None = None,
        config: dict[str, Any] | None = None,
        indication: str | None = None,
        phase: str | None = None,
        objective: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manager = self._manager()
        return manager.create_trial(
            trial_id=trial_id,
            config=config,
            indication=indication,
            phase=phase,
            objective=objective,
            status=status,
            metadata=metadata,
        )

    def update_trial(self, trial_id: str, *, updates: dict[str, Any]) -> dict[str, Any]:
        manager = self._manager()
        return manager.update_trial(trial_id, updates=updates)

    def remove_trial(self, trial_id: str) -> dict[str, Any]:
        manager = self._manager()
        return manager.remove_trial(trial_id)

    def enroll_patient(
        self,
        trial_id: str,
        *,
        patient_id: str | None = None,
        source: str | None = None,
        arm_id: str | None = None,
        demographics: dict[str, Any] | None = None,
        baseline: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manager = self._manager()
        return manager.enroll_patient(
            trial_id,
            patient_id=patient_id,
            source=source,
            arm_id=arm_id,
            demographics=demographics,
            baseline=baseline,
            metadata=metadata,
        )

    def enroll_simulated_patients(
        self,
        trial_id: str,
        *,
        count: int,
        seed: int | None = None,
    ) -> dict[str, Any]:
        manager = self._manager()
        return manager.enroll_simulated_patients(trial_id, count=count, seed=seed)

    def add_result(
        self,
        trial_id: str,
        *,
        patient_id: str,
        values: dict[str, Any],
        result_type: str = "endpoint",
        visit: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        manager = self._manager()
        return manager.record_result(
            trial_id,
            patient_id=patient_id,
            values=values,
            result_type=result_type,
            visit=visit,
            source=source,
        )

    def simulate_trial(
        self,
        trial_id: str,
        *,
        replicates: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        manager = self._manager()
        return manager.simulate_trial(trial_id, replicates=replicates, seed=seed)

    def _manager(self) -> Any:
        trial_mod = self._import_refua_clinical_module("refua_clinical.trial_management")

        if self._store_path is not None:
            store = self._store_path
        else:
            store = trial_mod.default_trial_store_path(base_dir=self._workspace_root)

        return trial_mod.ClinicalTrialManager(store)

    def _import_refua_clinical_module(self, module_name: str) -> Any:
        self._ensure_paths()
        return importlib.import_module(module_name)

    def _ensure_paths(self) -> None:
        if self._paths_ready:
            return

        candidates = [
            self._workspace_root / "refua-clinical" / "src",
            self._workspace_root / "ClawCures" / "src",
        ]
        for path in candidates:
            if path.exists():
                text = str(path)
                if text not in sys.path:
                    sys.path.insert(0, text)

        self._paths_ready = True
