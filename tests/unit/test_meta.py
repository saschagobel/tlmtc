"""Tests for tlmtc.meta."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tlmtc.data_contracts import InputMode
from tlmtc.meta import TrainRunMeta, read_run_meta, write_run_meta


@pytest.fixture
def train_meta() -> TrainRunMeta:
    """Create representative training-run metadata."""
    return TrainRunMeta(
        run_id="run123",
        target_name="Issue Type",
        checkpoint="microsoft/deberta-v3-base",
        proxy_checkpoint="microsoft/deberta-v3-small",
        sequence_length=128,
        trust_remote_code=True,
        input_mode=InputMode.SINGLE_TEXT,
        label_names=["routing", "compliance"],
        threshold_type="label",
        thresholds=[0.42, 0.61],
        transfer_learning=True,
        hyperparameter_tuning=True,
        threshold_optimization=True,
        scale_learning_rate=False,
        wrap_peft=True,
    )


class TestTrainRunMeta:
    """Test suite for TrainRunMeta."""

    def test_creates_expected_metadata(self) -> None:
        """Ensure metadata preserves resolved training-run state."""
        meta = TrainRunMeta(
            run_id="run123",
            target_name="Issue Type",
            checkpoint="microsoft/deberta-v3-base",
            proxy_checkpoint="microsoft/deberta-v3-small",
            sequence_length=256,
            trust_remote_code=True,
            input_mode=InputMode.PAIRED_TEXT,
            label_names=["routing", "compliance"],
            threshold_type="global",
            thresholds=[0.5],
            transfer_learning=True,
            hyperparameter_tuning=False,
            threshold_optimization=False,
            scale_learning_rate=True,
            wrap_peft=False,
        )

        assert meta.run_id == "run123"
        assert meta.target_name == "Issue Type"
        assert meta.checkpoint == "microsoft/deberta-v3-base"
        assert meta.proxy_checkpoint == "microsoft/deberta-v3-small"
        assert meta.sequence_length == 256
        assert meta.trust_remote_code is True
        assert meta.input_mode is InputMode.PAIRED_TEXT
        assert meta.label_names == ["routing", "compliance"]
        assert meta.threshold_type == "global"
        assert meta.thresholds == [0.5]
        assert meta.transfer_learning is True
        assert meta.hyperparameter_tuning is False
        assert meta.threshold_optimization is False
        assert meta.scale_learning_rate is True
        assert meta.wrap_peft is False
        assert meta.model_backends == ["torch"]

    def test_accepts_onnx_model_backend(self, train_meta: TrainRunMeta) -> None:
        """Ensure metadata can record ONNX export availability."""
        data = train_meta.model_dump(mode="python")
        data["model_backends"] = ["torch", "onnx"]

        meta = TrainRunMeta.model_validate(data)

        assert meta.model_backends == ["torch", "onnx"]

    def test_defaults_model_backends_for_existing_metadata(self, train_meta: TrainRunMeta) -> None:
        """Ensure metadata missing model_backends remains readable."""
        data = train_meta.model_dump(mode="python")
        data.pop("model_backends")

        meta = TrainRunMeta.model_validate(data)

        assert meta.model_backends == ["torch"]

    def test_accepts_missing_label_names(self, train_meta: TrainRunMeta) -> None:
        """Ensure metadata supports training runs without evaluation-derived labels."""
        data = train_meta.model_dump(mode="python")
        data["label_names"] = None

        meta = TrainRunMeta.model_validate(data)

        assert meta.label_names is None

    def test_created_at_defaults_to_timezone_aware_utc_datetime(self, train_meta: TrainRunMeta) -> None:
        """Ensure metadata timestamps are timezone-aware UTC datetimes."""
        assert isinstance(train_meta.created_at, datetime)
        assert train_meta.created_at.tzinfo is not None
        assert train_meta.created_at.utcoffset() == UTC.utcoffset(train_meta.created_at)

    @pytest.mark.parametrize(
        ("field", "invalid_value"),
        [
            ("sequence_length", 0),
            ("sequence_length", -1),
            ("threshold_type", "invalid_mode"),
            ("model_backends", ["torch", "invalid_backend"]),
        ],
    )
    def test_rejects_invalid_values(self, train_meta: TrainRunMeta, field: str, invalid_value: Any) -> None:
        """Ensure metadata fields enforce their declared constraints."""
        data = train_meta.model_dump(mode="python")
        data[field] = invalid_value

        with pytest.raises(ValidationError):
            TrainRunMeta.model_validate(data)

    def test_rejects_extra_fields(self, train_meta: TrainRunMeta) -> None:
        """Ensure persisted metadata has a strict schema."""
        data = train_meta.model_dump(mode="python")
        data["unexpected"] = "bad"

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            TrainRunMeta.model_validate(data)

    def test_is_frozen(self, train_meta: TrainRunMeta) -> None:
        """Ensure metadata objects are immutable after validation."""
        with pytest.raises(ValidationError):
            train_meta.run_id = "other"  # type: ignore[misc]


class TestTrainRunMetaIO:
    """Test suite for training metadata JSON IO."""

    def test_write_and_read_run_meta_roundtrips_metadata(self, tmp_path: Path, train_meta: TrainRunMeta) -> None:
        """Ensure metadata written to JSON can be read back unchanged."""
        path = tmp_path / "run_meta.json"

        write_run_meta(meta=train_meta, path=path)
        restored = read_run_meta(path)

        assert restored == train_meta

    def test_write_run_meta_does_not_create_parent_directory(self, tmp_path: Path, train_meta: TrainRunMeta) -> None:
        """Ensure metadata writing relies on resolved artifact directories already existing."""
        path = tmp_path / "missing" / "run_meta.json"

        with pytest.raises(FileNotFoundError):
            write_run_meta(meta=train_meta, path=path)

    def test_read_run_meta_rejects_invalid_json_contract(self, tmp_path: Path) -> None:
        """Ensure invalid metadata JSON fails validation."""
        path = tmp_path / "run_meta.json"
        path.write_text(
            """
            {
                "run_id": "run123",
                "created_at": "2026-05-06T20:00:00+00:00",
                "target_name": "Issue Type",
                "checkpoint": "checkpoint",
                "proxy_checkpoint": "proxy",
                "sequence_length": 0,
                "input_mode": "single_text",
                "label_names": ["routing"],
                "threshold_type": "label",
                "thresholds": [0.5],
                "transfer_learning": true,
                "hyperparameter_tuning": false,
                "threshold_optimization": true,
                "scale_learning_rate": false,
                "wrap_peft": false
            }
            """,
            encoding="utf-8",
        )

        with pytest.raises(ValidationError):
            read_run_meta(path)
