"""DuckDB-backed experiment catalog for persistent cross-run storage.

Accumulates experiment results and feedback decisions across pipeline runs.
Reuses the DuckDB engine from ``src/adapters/external_memory.py``.

Reference: Raasveldt & Muhleisen, "DuckDB: an Embeddable Analytical
Database", SIGMOD 2019.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_EXPERIMENTS_DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    dataset TEXT NOT NULL,
    endpoint TEXT,
    model_type TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    hyperparams TEXT,
    metrics TEXT,
    train_accuracy REAL,
    test_accuracy REAL,
    f1_score REAL,
    overfit_gap REAL,
    feedback_applied TEXT,
    dataset_characteristics TEXT,
    embedding_key TEXT,
    duration_seconds REAL,
    surrogate_prediction REAL,
    surrogate_uncertainty REAL
);
"""

_FEEDBACK_DDL = """
CREATE TABLE IF NOT EXISTS feedback_history (
    feedback_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    experiment_id TEXT,
    hypothesis_id TEXT NOT NULL,
    hypothesis_title TEXT,
    rationale TEXT,
    expected_impact TEXT,
    approved INTEGER NOT NULL,
    executed INTEGER NOT NULL,
    outcome_experiment_id TEXT,
    outcome_delta_f1 REAL,
    outcome_delta_accuracy REAL,
    confidence_score REAL DEFAULT 0.5
);
"""

_META_FEATURES_DDL = """
CREATE TABLE IF NOT EXISTS meta_features (
    dataset TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    value REAL,
    timestamp TEXT NOT NULL,
    PRIMARY KEY (dataset, feature_name)
);
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_exp_dataset ON experiments(dataset);",
    "CREATE INDEX IF NOT EXISTS idx_exp_model ON experiments(model_type);",
    "CREATE INDEX IF NOT EXISTS idx_exp_config ON experiments(config_hash);",
    "CREATE INDEX IF NOT EXISTS idx_exp_f1 ON experiments(f1_score);",
    "CREATE INDEX IF NOT EXISTS idx_fb_hypothesis ON feedback_history(hypothesis_id);",
    "CREATE INDEX IF NOT EXISTS idx_mf_dataset ON meta_features(dataset);",
]


class ExperimentCatalog:
    """Persistent experiment catalog backed by DuckDB.

    Parameters
    ----------
    db_path : path to the DuckDB database file.
        Use ``":memory:"`` for in-memory testing.
    """

    def __init__(self, db_path: str | Path = "data/experiment_catalog.db") -> None:
        import duckdb

        self._db_path = str(db_path)
        self._conn = duckdb.connect(self._db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(_EXPERIMENTS_DDL)
        self._conn.execute(_FEEDBACK_DDL)
        self._conn.execute(_META_FEATURES_DDL)
        for idx in _INDEXES:
            self._conn.execute(idx)
        # Migrate existing databases: add surrogate columns if missing
        self._maybe_add_column("experiments", "surrogate_prediction", "REAL")
        self._maybe_add_column("experiments", "surrogate_uncertainty", "REAL")

    def _maybe_add_column(self, table: str, column: str, dtype: str) -> None:
        """Add a column if it does not already exist (schema migration)."""
        cols = self._conn.execute(
            f"SELECT column_name FROM information_schema.columns "  # noqa: S608
            f"WHERE table_name = '{table}'"
        ).fetchdf()
        if column not in cols["column_name"].values:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {dtype}")  # noqa: S608

    def close(self) -> None:
        self._conn.close()

    # ── Experiment CRUD ─────────────────────────────────────────────

    def record_experiment(
        self,
        *,
        dataset: str,
        model_type: str,
        config_hash: str,
        endpoint: str | None = None,
        hyperparams: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        train_accuracy: float | None = None,
        test_accuracy: float | None = None,
        f1_score: float | None = None,
        overfit_gap: float | None = None,
        feedback_applied: list[str] | None = None,
        dataset_characteristics: dict[str, Any] | None = None,
        embedding_key: str | None = None,
        duration_seconds: float | None = None,
        surrogate_prediction: float | None = None,
        surrogate_uncertainty: float | None = None,
        experiment_id: str | None = None,
        timestamp: str | None = None,
    ) -> str:
        """Insert a new experiment row. Returns the experiment_id."""
        eid = experiment_id or str(uuid.uuid4())
        ts = timestamp or datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO experiments (
                experiment_id, timestamp, dataset, endpoint, model_type,
                config_hash, hyperparams, metrics, train_accuracy,
                test_accuracy, f1_score, overfit_gap, feedback_applied,
                dataset_characteristics, embedding_key, duration_seconds,
                surrogate_prediction, surrogate_uncertainty
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                eid,
                ts,
                dataset,
                endpoint,
                model_type,
                config_hash,
                json.dumps(hyperparams) if hyperparams else None,
                json.dumps(metrics) if metrics else None,
                train_accuracy,
                test_accuracy,
                f1_score,
                overfit_gap,
                json.dumps(feedback_applied) if feedback_applied else None,
                json.dumps(dataset_characteristics) if dataset_characteristics else None,
                embedding_key,
                duration_seconds,
                surrogate_prediction,
                surrogate_uncertainty,
            ],
        )
        logger.info("Recorded experiment %s (%s / %s)", eid[:8], dataset, model_type)
        return eid

    def record_feedback(
        self,
        *,
        experiment_id: str | None = None,
        hypothesis_id: str,
        hypothesis_title: str | None = None,
        rationale: str | None = None,
        expected_impact: str | None = None,
        approved: bool,
        executed: bool,
        feedback_id: str | None = None,
        timestamp: str | None = None,
        confidence_score: float = 0.5,
    ) -> str:
        """Insert a feedback history row. Returns the feedback_id."""
        fid = feedback_id or str(uuid.uuid4())
        ts = timestamp or datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO feedback_history (
                feedback_id, timestamp, experiment_id, hypothesis_id,
                hypothesis_title, rationale, expected_impact,
                approved, executed, confidence_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                fid,
                ts,
                experiment_id,
                hypothesis_id,
                hypothesis_title,
                rationale,
                expected_impact,
                1 if approved else 0,
                1 if executed else 0,
                confidence_score,
            ],
        )
        logger.info("Recorded feedback %s (hypothesis=%s)", fid[:8], hypothesis_id)
        return fid

    def link_feedback_outcome(
        self,
        feedback_id: str,
        outcome_experiment_id: str,
    ) -> None:
        """Link a feedback action to the subsequent experiment run.

        Computes delta metrics between the original experiment that
        triggered the feedback and the outcome experiment.
        """
        # Fetch the feedback row to find the baseline experiment
        fb_row = self._conn.execute(
            "SELECT experiment_id FROM feedback_history WHERE feedback_id = ?",
            [feedback_id],
        ).fetchone()
        if fb_row is None:
            raise ValueError(f"Feedback {feedback_id} not found")

        baseline_id = fb_row[0]
        delta_f1: float | None = None
        delta_acc: float | None = None

        if baseline_id:
            baseline = self._conn.execute(
                "SELECT f1_score, test_accuracy FROM experiments WHERE experiment_id = ?",
                [baseline_id],
            ).fetchone()
            outcome = self._conn.execute(
                "SELECT f1_score, test_accuracy FROM experiments WHERE experiment_id = ?",
                [outcome_experiment_id],
            ).fetchone()
            if baseline and outcome:
                if baseline[0] is not None and outcome[0] is not None:
                    delta_f1 = outcome[0] - baseline[0]
                if baseline[1] is not None and outcome[1] is not None:
                    delta_acc = outcome[1] - baseline[1]

        self._conn.execute(
            """
            UPDATE feedback_history
            SET outcome_experiment_id = ?,
                outcome_delta_f1 = ?,
                outcome_delta_accuracy = ?
            WHERE feedback_id = ?
            """,
            [outcome_experiment_id, delta_f1, delta_acc, feedback_id],
        )
        logger.info(
            "Linked feedback %s -> experiment %s (delta_f1=%s)",
            feedback_id[:8],
            outcome_experiment_id[:8],
            delta_f1,
        )

    # ── Queries ─────────────────────────────────────────────────────

    def query(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        """Execute raw SQL and return a DataFrame."""
        if params:
            return self._conn.execute(sql, params).fetchdf()
        return self._conn.execute(sql).fetchdf()

    def get_experiments(
        self,
        dataset: str | None = None,
        model_type: str | None = None,
        min_f1: float | None = None,
    ) -> pd.DataFrame:
        """Filtered retrieval of experiments."""
        clauses: list[str] = []
        params: list[Any] = []
        if dataset is not None:
            clauses.append("dataset = ?")
            params.append(dataset)
        if model_type is not None:
            clauses.append("model_type = ?")
            params.append(model_type)
        if min_f1 is not None:
            clauses.append("f1_score >= ?")
            params.append(min_f1)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM experiments{where} ORDER BY timestamp DESC"
        return self._conn.execute(sql, params).fetchdf()

    def get_feedback(
        self,
        hypothesis_id: str | None = None,
        approved_only: bool = False,
    ) -> pd.DataFrame:
        """Filtered retrieval of feedback history."""
        clauses: list[str] = []
        params: list[Any] = []
        if hypothesis_id is not None:
            clauses.append("hypothesis_id = ?")
            params.append(hypothesis_id)
        if approved_only:
            clauses.append("approved = 1")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM feedback_history{where} ORDER BY timestamp DESC"
        return self._conn.execute(sql, params).fetchdf()

    # ── Import & Summary ────────────────────────────────────────────

    def import_existing_registry(self, registry_csv_path: str | Path) -> int:
        """One-time migration of results/registry.csv into the catalog.

        Returns the number of rows imported.
        """
        path = Path(registry_csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Registry CSV not found: {path}")

        df = pd.read_csv(path)
        imported = 0
        for _, row in df.iterrows():
            metrics_dict: dict[str, Any] = {}
            for col in ["precision", "recall", "f1_score", "test_accuracy"]:
                if col in row and pd.notna(row[col]):
                    metrics_dict[col] = float(row[col])

            hyperparams_dict: dict[str, Any] = {}
            if "n_estimators" in row and pd.notna(row["n_estimators"]):
                hyperparams_dict["n_estimators"] = int(row["n_estimators"])

            self.record_experiment(
                dataset=str(row.get("dataset", "unknown")),
                model_type=str(row.get("model_type", "unknown")),
                config_hash=str(row.get("config_hash", "unknown")),
                hyperparams=hyperparams_dict if hyperparams_dict else None,
                metrics=metrics_dict if metrics_dict else None,
                train_accuracy=float(row["train_accuracy"]) if pd.notna(row.get("train_accuracy")) else None,
                test_accuracy=float(row["test_accuracy"]) if pd.notna(row.get("test_accuracy")) else None,
                f1_score=float(row["f1_score"]) if pd.notna(row.get("f1_score")) else None,
                overfit_gap=float(row["overfit_gap"]) if pd.notna(row.get("overfit_gap")) else None,
            )
            imported += 1
        logger.info("Imported %d rows from %s", imported, path)
        return imported

    def summary(self) -> dict[str, Any]:
        """Quick stats: total experiments, unique datasets, best model per dataset."""
        total = self._conn.execute("SELECT COUNT(*) FROM experiments").fetchone()
        total_count = total[0] if total else 0

        datasets = self._conn.execute("SELECT DISTINCT dataset FROM experiments").fetchdf()
        unique_datasets = datasets["dataset"].tolist() if not datasets.empty else []

        best_per_dataset: dict[str, dict[str, Any]] = {}
        for ds in unique_datasets:
            row = self._conn.execute(
                """
                SELECT model_type, f1_score, config_hash
                FROM experiments
                WHERE dataset = ? AND f1_score IS NOT NULL
                ORDER BY f1_score DESC
                LIMIT 1
                """,
                [ds],
            ).fetchone()
            if row:
                best_per_dataset[ds] = {
                    "model_type": row[0],
                    "f1_score": row[1],
                    "config_hash": row[2],
                }

        feedback_count = self._conn.execute("SELECT COUNT(*) FROM feedback_history").fetchone()
        total_feedback = feedback_count[0] if feedback_count else 0

        return {
            "total_experiments": total_count,
            "unique_datasets": unique_datasets,
            "n_datasets": len(unique_datasets),
            "best_per_dataset": best_per_dataset,
            "total_feedback": total_feedback,
        }

    # ── Meta-Features (B.6.1) ───────────────────────────────────────

    def record_meta_features(
        self,
        dataset: str,
        features: dict[str, float],
        timestamp: str | None = None,
    ) -> int:
        """Store computed meta-features for a dataset.

        Uses INSERT OR REPLACE so re-extraction overwrites stale values.
        Returns the number of features stored.
        """
        ts = timestamp or datetime.now(UTC).isoformat()
        stored = 0
        for name, value in features.items():
            self._conn.execute(
                "INSERT OR REPLACE INTO meta_features (dataset, feature_name, value, timestamp) "
                "VALUES (?, ?, ?, ?)",
                [dataset, name, float(value) if value is not None else None, ts],
            )
            stored += 1
        logger.info("Stored %d meta-features for dataset %s", stored, dataset)
        return stored

    def get_meta_features(self, dataset: str) -> dict[str, float]:
        """Retrieve meta-features for a single dataset.

        Returns
        -------
        dict mapping feature_name to value. Empty dict if none stored.
        """
        df = self._conn.execute(
            "SELECT feature_name, value FROM meta_features WHERE dataset = ?",
            [dataset],
        ).fetchdf()
        if df.empty:
            return {}
        return dict(zip(df["feature_name"], df["value"]))

    def get_all_meta_features(self) -> pd.DataFrame:
        """Retrieve meta-features for ALL datasets as a pivot table.

        Returns
        -------
        DataFrame with datasets as rows, feature names as columns.
        Index = dataset name. Empty DataFrame if no data.
        """
        df = self._conn.execute(
            "SELECT dataset, feature_name, value FROM meta_features"
        ).fetchdf()
        if df.empty:
            return pd.DataFrame()
        return df.pivot(index="dataset", columns="feature_name", values="value")
