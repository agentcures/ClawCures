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
        self._workspace_root = (
            workspace_root.resolve() if workspace_root else _workspace_root_from_file()
        )
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
        site_id: str | None = None,
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
            site_id=site_id,
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
        site_id: str | None = None,
    ) -> dict[str, Any]:
        manager = self._manager()
        return manager.record_result(
            trial_id,
            patient_id=patient_id,
            values=values,
            result_type=result_type,
            visit=visit,
            source=source,
            site_id=site_id,
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

    def list_sites(self, trial_id: str) -> dict[str, Any]:
        manager = self._manager()
        return manager.list_sites(trial_id)

    def upsert_site(
        self,
        trial_id: str,
        *,
        site_id: str,
        name: str | None = None,
        country_id: str | None = None,
        status: str | None = None,
        principal_investigator: str | None = None,
        target_enrollment: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manager = self._manager()
        return manager.upsert_site(
            trial_id,
            site_id=site_id,
            name=name,
            country_id=country_id,
            status=status,
            principal_investigator=principal_investigator,
            target_enrollment=target_enrollment,
            metadata=metadata,
        )

    def record_screening(
        self,
        trial_id: str,
        *,
        site_id: str,
        patient_id: str | None = None,
        status: str | None = None,
        arm_id: str | None = None,
        source: str | None = None,
        failure_reason: str | None = None,
        demographics: dict[str, Any] | None = None,
        baseline: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        auto_enroll: bool = False,
    ) -> dict[str, Any]:
        manager = self._manager()
        return manager.record_screening(
            trial_id,
            site_id=site_id,
            patient_id=patient_id,
            status=status,
            arm_id=arm_id,
            source=source,
            failure_reason=failure_reason,
            demographics=demographics,
            baseline=baseline,
            metadata=metadata,
            auto_enroll=auto_enroll,
        )

    def record_monitoring_visit(
        self,
        trial_id: str,
        *,
        site_id: str,
        visit_type: str | None = None,
        findings: list[str] | None = None,
        action_items: list[Any] | None = None,
        risk_score: float | None = None,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manager = self._manager()
        return manager.record_monitoring_visit(
            trial_id,
            site_id=site_id,
            visit_type=visit_type,
            findings=findings,
            action_items=action_items,
            risk_score=risk_score,
            outcome=outcome,
            metadata=metadata,
        )

    def add_query(
        self,
        trial_id: str,
        *,
        patient_id: str | None = None,
        site_id: str | None = None,
        field_name: str | None = None,
        description: str,
        status: str | None = None,
        severity: str | None = None,
        assignee: str | None = None,
        due_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manager = self._manager()
        return manager.add_query(
            trial_id,
            patient_id=patient_id,
            site_id=site_id,
            field_name=field_name,
            description=description,
            status=status,
            severity=severity,
            assignee=assignee,
            due_at=due_at,
            metadata=metadata,
        )

    def update_query(
        self,
        trial_id: str,
        *,
        query_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        manager = self._manager()
        return manager.update_query(
            trial_id,
            query_id=query_id,
            updates=updates,
        )

    def add_deviation(
        self,
        trial_id: str,
        *,
        description: str,
        site_id: str | None = None,
        patient_id: str | None = None,
        category: str | None = None,
        severity: str | None = None,
        status: str | None = None,
        corrective_action: str | None = None,
        preventive_action: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manager = self._manager()
        return manager.record_deviation(
            trial_id,
            description=description,
            site_id=site_id,
            patient_id=patient_id,
            category=category,
            severity=severity,
            status=status,
            corrective_action=corrective_action,
            preventive_action=preventive_action,
            metadata=metadata,
        )

    def add_safety_event(
        self,
        trial_id: str,
        *,
        patient_id: str,
        event_term: str,
        site_id: str | None = None,
        seriousness: str | None = None,
        expected: bool | None = None,
        relatedness: str | None = None,
        outcome: str | None = None,
        action_taken: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manager = self._manager()
        return manager.record_safety_event(
            trial_id,
            patient_id=patient_id,
            event_term=event_term,
            site_id=site_id,
            seriousness=seriousness,
            expected=expected,
            relatedness=relatedness,
            outcome=outcome,
            action_taken=action_taken,
            metadata=metadata,
        )

    def upsert_milestone(
        self,
        trial_id: str,
        *,
        milestone_id: str | None = None,
        name: str | None = None,
        target_date: str | None = None,
        status: str | None = None,
        owner: str | None = None,
        actual_date: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manager = self._manager()
        return manager.upsert_milestone(
            trial_id,
            milestone_id=milestone_id,
            name=name,
            target_date=target_date,
            status=status,
            owner=owner,
            actual_date=actual_date,
            metadata=metadata,
        )

    def operations_snapshot(self, trial_id: str) -> dict[str, Any]:
        manager = self._manager()
        return manager.operations_snapshot(trial_id)

    def _manager(self) -> Any:
        trial_mod = self._import_refua_clinical_module(
            "refua_clinical.trial_management"
        )

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
