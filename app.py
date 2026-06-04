import gc
import json
import os
import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
from datasets import DatasetDict
from dotenv import load_dotenv
import torch

load_dotenv()

from open_autonlu.auto_classes import (  # noqa: E402
    TextClassificationInferenceManager,
    TextClassificationTrainingPipeline,
    TokenClassificationTrainingPipeline,
    TokenClassificationInferenceManager,
)

from open_autonlu.data.utils import convert_bio_to_spans  # noqa: E402
from open_autonlu.methods.data_types import (  # noqa: E402
    ClassificationEvaluationResult,
    SaveFormat,
    OodMethod,
    TrainingArtifactInfo,
)

import plotly.express as px  # noqa: E402

DATA_DIR = "data"
MODEL_BASE_DIR = "saved_models/streamlit_demo"
MAX_INFERENCE_BATCH_SIZE = 20000

_DEFAULT_TRAINING_CONFIG: Dict[str, Any] = {
    "ood_method": "AUTO",
    "threshold_factor": 1.0,
    "llm_augmentation": False,
    "llm_test_generation": False,
}


def _default_llm_config() -> Dict[str, str]:
    return {
        "api_key": os.environ.get("MODEL_API_KEY", ""),
        "base_url": os.environ.get("BASE_URL", ""),
        "model_id": os.environ.get("MODEL_ID", ""),
    }


# Process scheduler state shared across all reruns
@st.cache_resource
def _shared_scheduler_state():
    return {
        "heavy_op_lock": threading.Lock(),
        "queue_lock": threading.Lock(),
        "queue": deque(),
    }


_SCHED = _shared_scheduler_state()
_heavy_op_lock: threading.Lock = _SCHED["heavy_op_lock"]
_queue_lock: threading.Lock = _SCHED["queue_lock"]
_heavy_op_queue: deque = _SCHED["queue"]

_QUEUE_REFRESH_SECONDS = 1


_SESSION_META_FILENAME = "session_meta.json"


def _session_id() -> str:
    if "session_id" in st.session_state:
        return st.session_state.session_id
    sid = st.query_params.get("sid")
    if sid and len(sid) == 12 and all(c in "0123456789abcdef" for c in sid.lower()):
        st.session_state.session_id = sid
        return sid
    new_id = uuid.uuid4().hex[:12]
    st.session_state.session_id = new_id
    st.query_params["sid"] = new_id
    return new_id


def _current_queue_position() -> Optional[int]:
    sid = _session_id()
    with _queue_lock:
        for i, (s, _) in enumerate(_heavy_op_queue, 1):
            if s == sid:
                return i
    return None


def _session_data_dir() -> str:
    return os.path.join(DATA_DIR, _session_id())


def _session_model_dir() -> str:
    return os.path.join(MODEL_BASE_DIR, _session_id())


def get_train_path() -> str:
    return os.path.join(_session_data_dir(), "demo_train.csv")


def get_test_path() -> str:
    return os.path.join(_session_data_dir(), "demo_test.csv")


def get_ner_train_path() -> str:
    return os.path.join(_session_data_dir(), "demo_train.json")


def get_ner_test_path() -> str:
    return os.path.join(_session_data_dir(), "demo_test.json")


_PROJECT_ROOT = Path(__file__).resolve().parent
_SAMPLE_DATA_ROOT = _PROJECT_ROOT / "examples" / "test_data"
SAMPLE_TRAIN_TEXT = (
    _SAMPLE_DATA_ROOT / "noise_n_shot_data" / "snips_200_300_0.01_42_train.csv"
)
SAMPLE_TEST_TEXT = (
    _SAMPLE_DATA_ROOT / "noise_n_shot_data" / "snips_200_300_0.01_42_test.csv"
)
SAMPLE_TRAIN_NER = _SAMPLE_DATA_ROOT / "noise_n_shot_data_ner" / "train.json"
SAMPLE_TEST_NER = _SAMPLE_DATA_ROOT / "noise_n_shot_data_ner" / "test.json"

AGGREGATE_ROWS = frozenset(["accuracy", "macro avg", "weighted avg", "micro avg"])

TASK_TEXT_CLF = "Text Classification"
TASK_NER = "Named Entity Recognition"


def ensure_dirs():
    os.makedirs(_session_data_dir(), exist_ok=True)
    os.makedirs(_session_model_dir(), exist_ok=True)


def _session_meta_path() -> str:
    return os.path.join(_session_data_dir(), _SESSION_META_FILENAME)


def _update_meta_model_path(meta_path: str, model_path: str) -> None:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        meta = {}
    meta["model_path"] = model_path
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


_JOB_RUNNING_FILENAME = "job_running.json"


def _job_running_path() -> str:
    return os.path.join(_session_data_dir(), _JOB_RUNNING_FILENAME)


def _save_session_meta():
    ensure_dirs()
    meta = {
        "task_type": st.session_state.get("task_type", TASK_TEXT_CLF),
        "language": st.session_state.get("language", "en"),
        "tab_choice": st.session_state.get("tab_choice", "Data Input"),
        "model_path": st.session_state.get("model_path"),
        "training_config": st.session_state.get("training_config", {}),
        "llm_config": st.session_state.get("llm_config", {}),
    }
    with open(_session_meta_path(), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _load_session_meta() -> Optional[Dict[str, Any]]:
    path = _session_meta_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _find_latest_model_dir() -> Optional[str]:
    d = _session_model_dir()
    if not os.path.isdir(d):
        return None
    best_path = None
    best_mtime = 0
    for name in os.listdir(d):
        path = os.path.join(d, name)
        if os.path.isdir(path):
            m = os.path.getmtime(path)
            if m > best_mtime:
                best_mtime = m
                best_path = path
    return best_path


_TRAINING_RESULTS_FILENAME = "training_results.json"
_DQ_RESULTS_FILENAME = "dq_results.json"
_DQ_REPORT_FILENAME = "dq_aggregated_report.csv"


@dataclass
class _PersistedDqOutput:
    indices_to_remove: List[int]
    aggregated_report_path: Optional[str]
    splits_filtered: Any = None


def _save_training_results(
    artifact, method_name: str, path: Optional[str] = None
) -> None:
    if artifact is None:
        return
    if path is None:
        ensure_dirs()
        path = os.path.join(_session_data_dir(), _TRAINING_RESULTS_FILENAME)
    try:
        labels = getattr(artifact, "labels", []) or []
        payload = {
            "method_name": method_name or "unknown",
            "labels": labels,
        }
        if artifact.test_metrics is not None:
            cm = artifact.test_metrics.confusion_matrix
            cm_list = cm.tolist() if hasattr(cm, "tolist") else None
            payload["classification_report"] = getattr(
                artifact.test_metrics, "classification_report", {}
            )
            payload["f1"] = getattr(artifact.test_metrics, "f1", None)
            payload["confusion_matrix"] = cm_list
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def _load_training_results() -> Optional[tuple]:
    path = os.path.join(_session_data_dir(), _TRAINING_RESULTS_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        method_name = data.get("method_name", "unknown")
        labels = data.get("labels", [])
        classification_report = data.get("classification_report", {})
        f1 = data.get("f1")
        cm_list = data.get("confusion_matrix")
        test_metrics = None
        if classification_report or f1 is not None or cm_list:
            cm = np.array(cm_list) if cm_list else np.array([])
            test_metrics = ClassificationEvaluationResult(
                confusion_matrix=cm,
                classification_report=classification_report or {},
                f1=float(f1) if f1 is not None else 0.0,
            )
        artifact = TrainingArtifactInfo(
            hyperparameters={},
            labels=labels,
            dev_metrics=None,
            test_metrics=test_metrics,
            test_metrics_inscope=None,
        )
        return (artifact, method_name)
    except Exception:
        return None


def _save_dq_results(dq_output, data_dir: Optional[str] = None) -> None:
    if dq_output is None:
        return
    if data_dir is None:
        ensure_dirs()
        data_dir = _session_data_dir()
    indices = list(getattr(dq_output, "indices_to_remove", None) or [])
    src_report = getattr(dq_output, "aggregated_report_path", None)
    dst_report = os.path.join(data_dir, _DQ_REPORT_FILENAME)
    if src_report and os.path.isfile(src_report):
        try:
            shutil.copy2(src_report, dst_report)
        except Exception:
            dst_report = None
    else:
        dst_report = None
    path = os.path.join(data_dir, _DQ_RESULTS_FILENAME)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"indices_to_remove": indices, "aggregated_report_path": dst_report},
                f,
                indent=2,
            )
    except Exception:
        pass


def _load_dq_results() -> Optional[_PersistedDqOutput]:
    path = os.path.join(_session_data_dir(), _DQ_RESULTS_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        indices = data.get("indices_to_remove", [])
        report_path = data.get("aggregated_report_path")
        if report_path and not os.path.isfile(report_path):
            report_path = None
        return _PersistedDqOutput(
            indices_to_remove=indices,
            aggregated_report_path=report_path,
            splits_filtered=None,
        )
    except Exception:
        return None


def empty_cache():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def init_state():
    if "train_data" in st.session_state:
        _session_id()
        if st.session_state.get("training_artifact") is None:
            loaded = _load_training_results()
            if loaded is not None:
                st.session_state.training_artifact, st.session_state.method_name = (
                    loaded[0],
                    loaded[1],
                )
        if st.session_state.get("dq_output") is None:
            loaded_dq = _load_dq_results()
            if loaded_dq is not None:
                st.session_state.dq_output = loaded_dq
        if st.session_state.get("model_path") is None:
            meta = _load_session_meta()
            model_path = meta.get("model_path") if meta else None
            if model_path and os.path.isdir(model_path):
                st.session_state.model_path = model_path
            else:
                st.session_state.model_path = _find_latest_model_dir()
        return
    _session_id()
    meta = _load_session_meta()
    if meta is not None:
        # Restore data from disk
        st.session_state.task_type = meta.get("task_type", TASK_TEXT_CLF)
        st.session_state.language = meta.get("language", "en")
        st.session_state.tab_choice = meta.get("tab_choice", "Data Input")
        st.session_state.training_config = meta.get(
            "training_config", dict(_DEFAULT_TRAINING_CONFIG)
        )
        st.session_state.llm_config = meta.get("llm_config", _default_llm_config())
        model_path = meta.get("model_path")
        if model_path and os.path.isdir(model_path):
            st.session_state.model_path = model_path
        else:
            st.session_state.model_path = _find_latest_model_dir()
        job_running_path = _job_running_path()
        if os.path.isfile(job_running_path):
            try:
                with open(job_running_path, encoding="utf-8") as f:
                    job_info = json.load(f)
                action = job_info.get("action", "training")
                st.session_state.tab_choice = (
                    "Training" if action == "training" else "Data Quality"
                )
                st.session_state.training_in_progress = True
            except Exception:
                st.session_state.training_in_progress = False
        else:
            st.session_state.training_in_progress = False
        st.session_state.train_data = pd.DataFrame(columns=["text", "label"])
        st.session_state.test_data = None
        st.session_state.ner_train_data = None
        st.session_state.ner_test_data = None
        if os.path.isfile(get_train_path()):
            try:
                st.session_state.train_data = pd.read_csv(get_train_path())
                st.session_state.train_data = normalize_train_data(
                    st.session_state.train_data
                )
            except Exception:
                pass
        if os.path.isfile(get_test_path()):
            try:
                st.session_state.test_data = pd.read_csv(get_test_path())
                st.session_state.test_data = normalize_test_data(
                    st.session_state.test_data
                )
            except Exception:
                pass
        if os.path.isfile(get_ner_train_path()):
            try:
                with open(get_ner_train_path(), encoding="utf-8") as f:
                    st.session_state.ner_train_data = json.load(f)
            except Exception:
                pass
        if os.path.isfile(get_ner_test_path()):
            try:
                with open(get_ner_test_path(), encoding="utf-8") as f:
                    st.session_state.ner_test_data = json.load(f)
            except Exception:
                pass
        loaded_dq = _load_dq_results()
        st.session_state.dq_output = loaded_dq if loaded_dq is not None else None
        loaded_training = _load_training_results()
        if loaded_training is not None:
            st.session_state.training_artifact, st.session_state.method_name = (
                loaded_training[0],
                loaded_training[1],
            )
        else:
            st.session_state.training_artifact = None
            st.session_state.method_name = None
        st.session_state.batch_predictions = None
        return
    # Default empty state
    st.session_state.task_type = TASK_TEXT_CLF
    st.session_state.train_data = pd.DataFrame(columns=["text", "label"])
    st.session_state.test_data = None
    st.session_state.ner_train_data = None
    st.session_state.ner_test_data = None
    st.session_state.dq_output = None
    st.session_state.training_artifact = None
    st.session_state.method_name = None
    st.session_state.model_path = None
    st.session_state.batch_predictions = None
    st.session_state.training_config = dict(_DEFAULT_TRAINING_CONFIG)
    st.session_state.llm_config = _default_llm_config()
    st.session_state.language = "en"
    st.session_state.tab_choice = "Data Input"
    st.session_state.training_in_progress = False


def on_task_type_change():
    st.session_state.dq_output = None
    st.session_state.training_artifact = None
    st.session_state.method_name = None
    st.session_state.model_path = None
    st.session_state.batch_predictions = None
    _save_session_meta()


def is_ner() -> bool:
    return st.session_state.task_type == TASK_NER


# Data normalization


def normalize_train_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "text" not in df.columns or "label" not in df.columns:
        st.error("Train data should contain text and label fields")
        return None
    if "anc_label" in df.columns:
        anc_series = df["anc_label"]
        if anc_series.isna().all() or (anc_series.astype(str).str.strip() == "").all():
            df = df.drop(columns=["anc_label"])
    columns = ["text", "label"]
    if "anc_label" in df.columns:
        columns.append("anc_label")
    return df[columns]


def normalize_test_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "text" not in df.columns or "label" not in df.columns:
        st.error("Test data should contain text and label fields")
        return None
    return df[["text", "label"]]


# Data input callbacks


def add_train_file():
    if st.session_state.train_file is None:
        return
    df = pd.read_csv(st.session_state.train_file)
    df = normalize_train_data(df)
    if df is None:
        return
    st.session_state.train_data = df
    save_inputs()


def add_test_file():
    if st.session_state.test_file is None:
        return
    df = pd.read_csv(st.session_state.test_file)
    df = normalize_test_data(df)
    if df is None:
        return
    st.session_state.test_data = df
    save_inputs()


def add_ner_train_file():
    if st.session_state.ner_train_file is None:
        return
    try:
        content = json.load(st.session_state.ner_train_file)
        if not isinstance(content, list):
            st.error("JSON should contain a list of records")
            return
        st.session_state.ner_train_data = content
        save_ner_inputs()
    except json.JSONDecodeError:
        st.error("Invalid JSON file")


def add_ner_test_file():
    if st.session_state.ner_test_file is None:
        return
    try:
        content = json.load(st.session_state.ner_test_file)
        if not isinstance(content, list):
            st.error("JSON should contain a list of records")
            return
        st.session_state.ner_test_data = content
        save_ner_inputs()
    except json.JSONDecodeError:
        st.error("Invalid JSON file")


# Save helpers


def save_inputs():
    ensure_dirs()
    st.session_state.train_data.to_csv(get_train_path(), index=False)
    if st.session_state.test_data is not None:
        st.session_state.test_data.to_csv(get_test_path(), index=False)
    elif os.path.exists(get_test_path()):
        os.remove(get_test_path())
    _save_session_meta()


def save_ner_inputs():
    ensure_dirs()
    if st.session_state.ner_train_data is not None:
        with open(get_ner_train_path(), "w", encoding="utf-8") as f:
            json.dump(st.session_state.ner_train_data, f, ensure_ascii=False)
    if st.session_state.ner_test_data is not None:
        with open(get_ner_test_path(), "w", encoding="utf-8") as f:
            json.dump(st.session_state.ner_test_data, f, ensure_ascii=False)
    elif os.path.exists(get_ner_test_path()):
        os.remove(get_ner_test_path())
    _save_session_meta()


# NER data preview


def ner_data_summary(data: list) -> pd.DataFrame:
    rows = []
    for i, record in enumerate(data):
        text = record.get("text", "")
        spans = record.get("spans", [])
        entity_labels = sorted({s.get("label", "?") for s in spans}) if spans else []
        rows.append(
            {
                "#": i,
                "text": text[:80] + "..." if len(text) > 80 else text,
                "entities": len(spans),
                "labels": ", ".join(entity_labels) if entity_labels else "—",
            }
        )
    return pd.DataFrame(rows)


# Data Quality


def _run_dq_impl(job_path: str):
    if is_ner():
        _run_ner_dq(job_path)
    else:
        _run_text_clf_dq(job_path)


def run_dq():
    sid = _session_id()
    with _queue_lock:
        if any(s == sid for s, _ in _heavy_op_queue):
            return
        _heavy_op_queue.append((sid, "dq"))
    st.session_state.queued_action = "dq"
    st.session_state.tab_choice = "Data Quality"
    _save_session_meta()


def _run_text_clf_dq(job_path: str):
    if st.session_state.train_data.empty:
        st.error("Training data is missing")
        return
    save_inputs()
    _save_session_meta()
    lang = st.session_state.get("language", "en")
    train_path = get_train_path()
    test_path = get_test_path() if st.session_state.test_data is not None else None
    data_dir = _session_data_dir()
    ensure_dirs()

    worker_result = [None]
    worker_error = [None]

    def _worker():
        try:
            pipeline = TextClassificationTrainingPipeline(
                train_path=train_path,
                test_path=test_path,
                config_overrides={"language": lang},
            )
            dq_output = pipeline.diagnose()
            if dq_output is not None:
                _save_dq_results(dq_output, data_dir=data_dir)
            worker_result[0] = dq_output
        except Exception as e:
            worker_error[0] = e
        finally:
            _heavy_op_lock.release()
            try:
                if os.path.isfile(job_path):
                    os.remove(job_path)
            except Exception:
                pass

    t = threading.Thread(target=_worker)
    t.start()
    try:
        t.join()
    except BaseException:
        return

    if worker_error[0] is not None:
        st.error(f"Error during data quality check: {worker_error[0]}")
        st.session_state.dq_output = None
        empty_cache()
        return

    st.session_state.dq_output = worker_result[0]
    empty_cache()
    if worker_result[0] is None:
        st.warning(
            "Data quality check skipped: dataset is too small for quality evaluation."
        )


def _run_ner_dq(job_path: str):
    if st.session_state.ner_train_data is None:
        st.error("NER training data is missing")
        return
    save_ner_inputs()
    lang = st.session_state.get("language", "en")
    ner_train_path = get_ner_train_path()
    ner_test_path = (
        get_ner_test_path() if st.session_state.ner_test_data is not None else None
    )
    data_dir = _session_data_dir()
    ensure_dirs()

    worker_result = [None]
    worker_error = [None]

    def _worker():
        try:
            pipeline = TokenClassificationTrainingPipeline(
                train_path=ner_train_path,
                test_path=ner_test_path,
                config_overrides={"language": lang},
            )
            dq_output = pipeline.diagnose()
            if dq_output is not None:
                _save_dq_results(dq_output, data_dir=data_dir)
            worker_result[0] = dq_output
        except Exception as e:
            worker_error[0] = e
        finally:
            _heavy_op_lock.release()
            try:
                if os.path.isfile(job_path):
                    os.remove(job_path)
            except Exception:
                pass

    t = threading.Thread(target=_worker)
    t.start()
    try:
        t.join()
    except BaseException:
        return

    if worker_error[0] is not None:
        st.error(f"Error during data quality check: {worker_error[0]}")
        st.session_state.dq_output = None
        empty_cache()
        return

    st.session_state.dq_output = worker_result[0]
    empty_cache()
    if worker_result[0] is None:
        st.warning(
            "Data quality check skipped: dataset is too small for quality evaluation."
        )


def apply_dq_filter():
    dq_out = st.session_state.get("dq_output")
    if dq_out is None:
        return
    indices = set(getattr(dq_out, "indices_to_remove", None) or [])
    filt = dq_out.splits_filtered
    if isinstance(filt, DatasetDict) and "train" in filt:
        df = filt["train"].to_pandas()
    elif indices and not st.session_state.train_data.empty:
        df = st.session_state.train_data.drop(
            index=[i for i in indices if i < len(st.session_state.train_data)],
            errors="ignore",
        ).reset_index(drop=True)
    else:
        return
    if "text" not in df.columns or "label" not in df.columns:
        return
    cols = ["text", "label"]
    if "anc_label" in df.columns:
        cols.append("anc_label")
    st.session_state.train_data = df[cols].copy()
    ensure_dirs()
    st.session_state.train_data.to_csv(get_train_path(), index=False)
    st.session_state.dq_apply_success = True


def apply_dq_filter_ner():
    dq_out = st.session_state.get("dq_output")
    if dq_out is None:
        return
    indices = set(getattr(dq_out, "indices_to_remove", None) or [])
    filt = dq_out.splits_filtered
    if isinstance(filt, DatasetDict) and "train" in filt:
        train_ds = filt["train"]
        if (
            "text" not in train_ds.column_names
            or "tokens" not in train_ds.column_names
            or "labels" not in train_ds.column_names
        ):
            return
        records = []
        for row in train_ds:
            text = row["text"]
            tokens = row["tokens"]
            labels = row["labels"]
            spans = convert_bio_to_spans(text, tokens, labels)
            records.append({"text": text, "spans": spans})
    elif indices and st.session_state.ner_train_data:
        records = [
            r for i, r in enumerate(st.session_state.ner_train_data) if i not in indices
        ]
    else:
        return
    st.session_state.ner_train_data = records
    ensure_dirs()
    with open(get_ner_train_path(), "w", encoding="utf-8") as f:
        json.dump(st.session_state.ner_train_data, f, ensure_ascii=False)
    st.session_state.dq_apply_success = True


# Training


def _run_training_impl(job_path: str):
    if is_ner():
        _run_ner_training(job_path)
    else:
        _run_text_clf_training(job_path)


def run_training():
    sid = _session_id()
    with _queue_lock:
        if any(s == sid for s, _ in _heavy_op_queue):
            return
        _heavy_op_queue.append((sid, "training"))
    st.session_state.queued_action = "training"
    st.session_state.tab_choice = "Training"
    _save_session_meta()


def _get_llm_client_overrides() -> Optional[Dict[str, Any]]:
    llm_cfg = st.session_state.get("llm_config", {})
    api_key = (llm_cfg.get("api_key") or "").strip()
    base_url = (llm_cfg.get("base_url") or "").strip()
    if not api_key or not base_url:
        return None
    return {
        "LlmClientConfig": {
            "api_key": api_key,
            "base_url": base_url,
            "model_id": (llm_cfg.get("model_id") or "").strip() or None,
        }
    }


def _run_text_clf_training(job_path: str):
    if st.session_state.train_data.empty:
        st.error("Training data is missing")
        return
    save_inputs()

    config = st.session_state.training_config
    ood = OodMethod[config["ood_method"]]
    config_overrides: Dict[str, Any] = {
        "ood_method": ood,
        "language": st.session_state.get("language", "en"),
    }

    if ood not in (OodMethod.NONE, OodMethod.LOGIT):
        config_overrides["threshold_factor"] = config.get("threshold_factor", 1.0)

    if config.get("llm_augmentation", False):
        llm_overrides = _get_llm_client_overrides()
        if llm_overrides is None:
            st.error(
                "LLM Augmentation is enabled but API key and Base URL are not set. "
                "Configure them in the Training Configuration section."
            )
            return
        config_overrides["llm_augmentation"] = {
            "enabled": True,
            "use_domain_analysis": True,
            "config_overrides": llm_overrides,
        }

    if config.get("llm_test_generation", False):
        llm_overrides = _get_llm_client_overrides()
        if llm_overrides is None:
            st.error(
                "Test generation is enabled but API key and Base URL are not set. "
                "Configure them in the Training Configuration section."
            )
            return
        config_overrides["llm_test_generation"] = {
            "enabled": True,
            "num_samples_per_class": 100,
            "use_domain_analysis": True,
            "config_overrides": llm_overrides,
        }

    train_path = get_train_path()
    test_path = get_test_path() if st.session_state.test_data is not None else None
    model_path = os.path.join(_session_model_dir(), f"model_{int(time.time())}")
    results_path = os.path.join(_session_data_dir(), _TRAINING_RESULTS_FILENAME)
    meta_path = _session_meta_path()
    ensure_dirs()

    worker_error = [None]

    worker_error = [None]

    def _worker():
        try:
            pipeline = TextClassificationTrainingPipeline(
                train_path=train_path,
                test_path=test_path,
                config_overrides=config_overrides,
            )
            artifact = pipeline.train()
            method_name = pipeline.training_method_cls.__name__
            pipeline.save(model_path, SaveFormat.TORCH)
            _save_training_results(artifact, method_name, path=results_path)
            _update_meta_model_path(meta_path, model_path)
        except Exception as e:
            worker_error[0] = e
        finally:
            _heavy_op_lock.release()
            try:
                os.remove(job_path)
            except OSError:
                pass

    t = threading.Thread(target=_worker)
    t.start()
    try:
        t.join()
    except BaseException:
        return

    if worker_error[0] is not None:
        empty_cache()
        st.error(f"Error during training: {worker_error[0]}")
        return

    loaded = _load_training_results()
    if loaded is not None:
        st.session_state.training_artifact = loaded[0]
        st.session_state.method_name = loaded[1]
    st.session_state.model_path = model_path
    _save_session_meta()
    empty_cache()
    st.success("Training completed successfully!")


def _run_ner_training(job_path: str):
    if st.session_state.ner_train_data is None:
        st.error("NER training data is missing")
        return
    save_ner_inputs()
    _save_session_meta()
    lang = st.session_state.get("language", "en")
    ner_train_path = get_ner_train_path()
    ner_test_path = (
        get_ner_test_path() if st.session_state.ner_test_data is not None else None
    )
    model_path = os.path.join(_session_model_dir(), f"ner_model_{int(time.time())}")
    results_path = os.path.join(_session_data_dir(), _TRAINING_RESULTS_FILENAME)
    meta_path = _session_meta_path()
    ensure_dirs()

    worker_error = [None]

    def _worker():
        try:
            pipeline = TokenClassificationTrainingPipeline(
                train_path=ner_train_path,
                test_path=ner_test_path,
                config_overrides={"language": lang},
            )
            artifact = pipeline.train()
            method_name = pipeline.training_method_cls.__name__
            pipeline.save(model_path, SaveFormat.TORCH)
            _save_training_results(artifact, method_name, path=results_path)
            _update_meta_model_path(meta_path, model_path)
        except Exception as e:
            worker_error[0] = e
        finally:
            _heavy_op_lock.release()
            try:
                os.remove(job_path)
            except OSError:
                pass

    t = threading.Thread(target=_worker)
    t.start()
    try:
        t.join()
    except BaseException:
        return

    if worker_error[0] is not None:
        empty_cache()
        st.error(f"Error during training: {worker_error[0]}")
        return

    loaded = _load_training_results()
    if loaded is not None:
        st.session_state.training_artifact = loaded[0]
        st.session_state.method_name = loaded[1]
    st.session_state.model_path = model_path
    _save_session_meta()
    empty_cache()
    st.success("Training completed successfully!")


# Inference


def predict_texts(texts: List[str]):
    if st.session_state.model_path is None:
        st.error("Trained model is missing")
        return None
    if is_ner():
        return _predict_ner(texts)
    return _predict_text_clf(texts)


def _predict_text_clf(texts: List[str]):
    manager = TextClassificationInferenceManager(st.session_state.model_path)
    outputs = manager.predict(texts, batch_size=32, return_all_hypotheses=False)
    rows = []
    for output in outputs:
        best = output.most_probable
        rows.append({"text": output.text, "label": best.label, "score": best.score})
    return pd.DataFrame(rows)


def _predict_ner(texts: List[str]):
    manager = TokenClassificationInferenceManager(st.session_state.model_path)
    outputs = manager.predict(texts, batch_size=32)
    rows = []
    for output in outputs:
        for entity in output.labels:
            rows.append(
                {
                    "text": output.text[:60] + "..."
                    if len(output.text) > 60
                    else output.text,
                    "entity": entity.text,
                    "label": entity.label,
                    "start": entity.start,
                    "end": entity.end,
                    "score": round(entity.score, 4),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["text", "entity", "label", "start", "end", "score"]
        )
    return pd.DataFrame(rows)


# Cache


def clear_cache():
    data_dir = _session_data_dir()
    for name in (
        _TRAINING_RESULTS_FILENAME,
        _DQ_RESULTS_FILENAME,
        _DQ_REPORT_FILENAME,
    ):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except Exception:
                pass
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.cache_data.clear()
    st.cache_resource.clear()
    empty_cache()


# Display helpers


def display_training_results(tab, artifact, method_name: str):
    tab.subheader("Training Results")
    tab.write(f"**Method:** {method_name}")

    if artifact.labels:
        tab.write(f"**Number of classes:** {len(artifact.labels)}")

    if artifact.test_metrics is not None:
        _display_test_metrics(tab, artifact)
    else:
        tab.info("Test dataset is missing", icon="ℹ️")


def _display_test_metrics(tab, artifact):
    tab.subheader("Test Set Metrics")
    test_report = pd.DataFrame(artifact.test_metrics.classification_report).T

    col1, col2 = tab.columns(2)
    if "macro avg" in test_report.index and "f1-score" in test_report.columns:
        col1.metric(
            "F1-Score (Macro)", f"{test_report.loc['macro avg', 'f1-score']:.4f}"
        )
    if "weighted avg" in test_report.index and "f1-score" in test_report.columns:
        col2.metric(
            "F1-Score (Weighted)", f"{test_report.loc['weighted avg', 'f1-score']:.4f}"
        )

    class_labels = [label for label in test_report.index if label not in AGGREGATE_ROWS]

    if (
        artifact.test_metrics.confusion_matrix is not None
        and artifact.test_metrics.confusion_matrix.size > 0
    ):
        _display_confusion_matrix(
            tab, artifact.test_metrics.confusion_matrix, class_labels, artifact.labels
        )


def _display_confusion_matrix(
    tab, cm, class_labels: List[str], fallback_labels: Optional[List[str]]
):
    try:
        labels = class_labels if class_labels else fallback_labels
        if not labels:
            tab.warning("Cannot display confusion matrix: no labels available")
            return
        if len(labels) != cm.shape[0]:
            if fallback_labels and len(fallback_labels) == cm.shape[0]:
                labels = fallback_labels
            else:
                return

        tab.subheader("Confusion Matrix")
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=labels,
            y=labels,
            aspect="auto",
            color_continuous_scale="Blues",
        )
        cm_height = max(600, len(labels) * 25)
        fig_cm.update_layout(height=cm_height)
        tab.plotly_chart(fig_cm, use_container_width=True)
    except Exception as e:
        tab.warning(f"Could not display confusion matrix: {str(e)}")


# UI

init_state()


@st.fragment(run_every=timedelta(seconds=_QUEUE_REFRESH_SECONDS))
def _heavy_op_queue_processor():
    sid = _session_id()
    if st.session_state.get("training_in_progress"):
        if not os.path.isfile(_job_running_path()):
            st.session_state.training_in_progress = False
            meta = _load_session_meta()
            meta_model = meta.get("model_path") if meta else None
            if meta_model and os.path.isdir(meta_model):
                st.session_state.model_path = meta_model
            elif st.session_state.get("model_path") is None:
                st.session_state.model_path = _find_latest_model_dir()
            if st.session_state.get("training_artifact") is None:
                loaded = _load_training_results()
                if loaded is not None:
                    st.session_state.training_artifact = loaded[0]
                    st.session_state.method_name = loaded[1]
            if st.session_state.get("dq_output") is None:
                loaded_dq = _load_dq_results()
                if loaded_dq is not None:
                    st.session_state.dq_output = loaded_dq
            st.rerun(scope="app")
    with _queue_lock:
        if not _heavy_op_queue or _heavy_op_queue[0][0] != sid:
            return
        item = _heavy_op_queue.popleft()
    action = item[1]
    if not _heavy_op_lock.acquire(blocking=False):
        with _queue_lock:
            _heavy_op_queue.appendleft(item)
        return
    st.session_state.training_in_progress = True
    st.session_state.queued_action = None
    st.session_state.tab_choice = "Training" if action == "training" else "Data Quality"
    ensure_dirs()
    job_path = _job_running_path()
    try:
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump({"action": action}, f)
    except Exception:
        pass
    if action == "training":
        _run_training_impl(job_path)
    else:
        _run_dq_impl(job_path)
    try:
        st.session_state.training_in_progress = False
    except Exception:
        pass
    try:
        _save_session_meta()
        st.rerun(scope="app")
    except Exception:
        pass


_heavy_op_queue_processor()

st.title("OpenAutoNLU Pipeline")

col_task, col_lang = st.columns(2)
with col_task:
    st.selectbox(
        "Task Type",
        options=[TASK_TEXT_CLF, TASK_NER],
        key="task_type",
        on_change=on_task_type_change,
    )
with col_lang:
    st.selectbox(
        "Language",
        options=["ru", "en"],
        format_func=lambda x: "Russian" if x == "ru" else "English",
        key="language",
        on_change=_save_session_meta,
        help="Affects default model (ruBert / BERT) and tokenization.",
    )

_TAB_OPTIONS = ["Data Input", "Data Quality", "Training", "Inference", "Clear Cache"]
if "tab_choice" not in st.session_state:
    st.session_state.tab_choice = "Data Input"
st.radio(
    "Tab",
    options=_TAB_OPTIONS,
    key="tab_choice",
    horizontal=True,
    label_visibility="collapsed",
    on_change=_save_session_meta,
)

data_input = st.container()
data_quality_tab = st.container()
training_tab = st.container()
inference_tab = st.container()
cache_tab = st.container()


@st.fragment(run_every=timedelta(seconds=_QUEUE_REFRESH_SECONDS))
def _data_quality_tab_content():
    if st.session_state.get("training_in_progress"):
        st.info(
            "Data quality check in progress... The page will update when it finishes "
            f"(refresh every {_QUEUE_REFRESH_SECONDS} s)."
        )
    if is_ner():
        if st.session_state.ner_train_data is not None:
            st.write(f"Train data: **{len(st.session_state.ner_train_data)} records**")
        else:
            st.info("Upload NER training data first", icon="ℹ️")
    else:
        st.dataframe(st.session_state.train_data, width=600)

    if st.session_state.get("queued_action") == "dq":
        pos = _current_queue_position()
        pos_text = f" (position {pos})" if pos is not None else ""
        st.info(
            f"You are in the queue{pos_text}. "
            f"The queue is checked every {_QUEUE_REFRESH_SECONDS} s; the status box below will update. "
            "When it's your turn, the check will start automatically"
        )
    st.button("Check Data Quality", key="dq", on_click=run_dq)

    if st.session_state.dq_output is not None:
        dq = st.session_state.dq_output
        num_candidates = len(list(dq.indices_to_remove))
        st.caption(
            "Training uses the current training data. "
            "Data quality check only marks candidates for removal. "
            'To train on cleaned data, click "Apply cleanup".'
        )
        if num_candidates == 0:
            st.success("Data quality check completed. No problematic samples found!")
        else:
            st.write(f"Candidates for removal: {num_candidates}")
            if dq.aggregated_report_path and os.path.exists(dq.aggregated_report_path):
                report_df = pd.read_csv(dq.aggregated_report_path)
                st.dataframe(report_df)
            else:
                st.info("Detailed report is being generated...")

            if not is_ner():
                st.button("Apply cleanup", key="apply_dq", on_click=apply_dq_filter)
            else:
                st.button(
                    "Apply cleanup",
                    key="apply_dq_ner",
                    on_click=apply_dq_filter_ner,
                )
        if st.session_state.pop("dq_apply_success", False):
            st.success("Cleaned data applied. Training will use it from now on.")


def _training_tab_content():
    if st.session_state.get("training_in_progress"):
        with st.status("Training in progress...", state="running", expanded=True):
            st.info(
                "Model is training. The page will update when it finishes (refresh every "
                f"{_QUEUE_REFRESH_SECONDS} s)."
            )
    if st.session_state.get("queued_action") == "training":
        pos = _current_queue_position()
        pos_text = f" (position {pos})" if pos is not None else ""
        st.info(
            f"You are in the queue{pos_text}. "
            f"The queue is checked every {_QUEUE_REFRESH_SECONDS} s; the status box below will update. "
            "When it's your turn, training will start automatically "
        )
    if not is_ner():
        st.session_state["_training_config_expanded"] = True
        with st.expander(
            "⚙️ Training Configuration",
            expanded=st.session_state["_training_config_expanded"],
        ):
            ood_method = st.selectbox(
                "OOD Detection Method",
                options=[
                    "AUTO",
                    "NONE",
                    "LOGIT",
                    "MARGINAL_MAHALANOBIS_OOD",
                    "MSP_OOD",
                ],
                index=[
                    "AUTO",
                    "NONE",
                    "LOGIT",
                    "MARGINAL_MAHALANOBIS_OOD",
                    "MSP_OOD",
                ].index(st.session_state.training_config.get("ood_method", "AUTO")),
                help="Method for out-of-distribution detection. AUTO selects automatically based on dataset size.",
            )

            show_threshold = ood_method not in ("NONE", "LOGIT")
            threshold_factor = st.session_state.training_config.get(
                "threshold_factor", 1.0
            )
            if show_threshold:
                threshold_factor = st.slider(
                    "OOD Threshold Factor",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(threshold_factor),
                    step=0.01,
                    help="Multiplier for the OOD detection threshold. "
                    "Higher values = fewer samples detected as OOD (more conservative). "
                    "Default: 1.0",
                )

            llm_augmentation = st.checkbox(
                "LLM Augmentation",
                value=st.session_state.training_config.get("llm_augmentation", False),
                help="Adds synthetic examples for low-resource classes: classes with fewer than 81 examples "
                "are augmented up to 81 per class via LLM. Requires API key and Base URL below.",
            )
            llm_test_generation = st.checkbox(
                "Test generation",
                value=st.session_state.training_config.get(
                    "llm_test_generation", False
                ),
                help="When test set is not uploaded, generates 100 examples per class via LLM. Uses the same API key and Base URL.",
            )

            if "llm_config" not in st.session_state:
                st.session_state.llm_config = _default_llm_config()
            llm_cfg = st.session_state.llm_config

            if llm_augmentation or llm_test_generation:
                st.caption(
                    "LLM configuration (required when augmentation or test generation is enabled). "
                    "Augmentation: classes with < 81 examples are augmented up to 81 per class. "
                    "Test generation: 100 examples per class when test set is not provided."
                )
                llm_api_key = st.text_input(
                    "LLM API Key",
                    value=llm_cfg.get("api_key", ""),
                    type="password",
                    key="llm_api_key",
                    help="API key for the LLM service.",
                )
                llm_base_url = st.text_input(
                    "LLM Base URL",
                    value=llm_cfg.get("base_url", ""),
                    key="llm_base_url",
                    help="Base URL of the LLM API (e.g. https://api.openai.com/v1/). ",
                )
                llm_model_id = st.text_input(
                    "LLM Model ID",
                    value=llm_cfg.get("model_id", ""),
                    key="llm_model_id",
                    help="Model name/ID (e.g. gpt-4). ",
                )
                st.session_state.llm_config = {
                    "api_key": llm_api_key,
                    "base_url": llm_base_url.strip(),
                    "model_id": llm_model_id.strip(),
                }

            st.session_state.training_config = {
                "ood_method": ood_method,
                "threshold_factor": threshold_factor,
                "llm_augmentation": llm_augmentation,
                "llm_test_generation": llm_test_generation,
            }

    st.button("Start Training", key="training", on_click=run_training)
    if st.session_state.training_artifact is not None:
        st.success("Training is complete. You can proceed to the **Inference** tab.")
        artifact = st.session_state.training_artifact
        method_name = st.session_state.method_name or "unknown"
        display_training_results(st, artifact, method_name)


if st.session_state.tab_choice == "Data Input":
    with data_input:
        if is_ner():
            data_input.info(
                "JSON file: list of objects with fields `text` and `spans`.\n"
                "Each span: `{label, start, end}`",
                icon="ℹ️",
            )
            data_input.markdown(
                "<p style='font-weight: 700; font-size: 1.15em;'>Train data</p>",
                unsafe_allow_html=True,
            )
            data_input.file_uploader(
                "",
                type="json",
                key="ner_train_file",
                on_change=add_ner_train_file,
                label_visibility="collapsed",
            )
            data_input.markdown(
                "<p style='font-weight: 700; font-size: 1.15em;'>Test data(optional)</p>",
                unsafe_allow_html=True,
            )
            data_input.file_uploader(
                "",
                type="json",
                key="ner_test_file",
                on_change=add_ner_test_file,
                label_visibility="collapsed",
            )
            _dl_expander_ner = data_input.expander(
                "Download data (example format)", expanded=False
            )
            with _dl_expander_ner:
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    st.download_button(
                        "Download train.json",
                        data=SAMPLE_TRAIN_NER.read_bytes(),
                        file_name="train.json",
                        mime="application/json",
                        key="download_ner_train",
                    )
                with dcol2:
                    st.download_button(
                        "Download test.json",
                        data=SAMPLE_TEST_NER.read_bytes(),
                        file_name="test.json",
                        mime="application/json",
                        key="download_ner_test",
                    )
            _train_preview_ner = data_input.expander(
                "Train data: preview", expanded=False
            )
            with _train_preview_ner:
                if st.session_state.ner_train_data:
                    summary = ner_data_summary(st.session_state.ner_train_data)
                    st.caption(f"**{len(st.session_state.ner_train_data)} records**")
                    st.dataframe(summary, use_container_width=True)
                else:
                    st.caption("No train data loaded.")
            _test_preview_ner = data_input.expander(
                "Test data: preview", expanded=False
            )
            with _test_preview_ner:
                if st.session_state.ner_test_data:
                    summary_test = ner_data_summary(st.session_state.ner_test_data)
                    st.caption(f"**{len(st.session_state.ner_test_data)} records**")
                    st.dataframe(summary_test, use_container_width=True)
                else:
                    st.caption("No test data loaded")
        else:
            data_input.info(
                "CSV file should contain columns: text, label (optionally anc_label).",
                icon="ℹ️",
            )
            data_input.markdown(
                "<p style='font-weight: 700; font-size: 1.15em;'>Train data</p>",
                unsafe_allow_html=True,
            )
            data_input.file_uploader(
                "",
                type="csv",
                key="train_file",
                on_change=add_train_file,
                label_visibility="collapsed",
            )
            data_input.markdown(
                "<p style='font-weight: 700; font-size: 1.15em;'>Test data(optional)</p>",
                unsafe_allow_html=True,
            )
            data_input.file_uploader(
                "",
                type="csv",
                key="test_file",
                on_change=add_test_file,
                label_visibility="collapsed",
            )
            _dl_expander_csv = data_input.expander(
                "Download data (example format)", expanded=False
            )
            with _dl_expander_csv:
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    st.download_button(
                        "Download train.csv",
                        data=SAMPLE_TRAIN_TEXT.read_bytes(),
                        file_name="train.csv",
                        mime="text/csv",
                        key="download_train",
                    )
                with dcol2:
                    st.download_button(
                        "Download test.csv",
                        data=SAMPLE_TEST_TEXT.read_bytes(),
                        file_name="test.csv",
                        mime="text/csv",
                        key="download_test",
                    )
            _train_preview = data_input.expander("Train data: preview", expanded=False)
            with _train_preview:
                if not st.session_state.train_data.empty:
                    st.caption(f"**{len(st.session_state.train_data)} rows**")
                    st.dataframe(st.session_state.train_data, use_container_width=True)
                else:
                    st.caption("No train data loaded.")
            _test_preview = data_input.expander("Test data: preview", expanded=False)
            with _test_preview:
                if (
                    st.session_state.test_data is not None
                    and not st.session_state.test_data.empty
                ):
                    st.caption(f"**{len(st.session_state.test_data)} rows**")
                    st.dataframe(st.session_state.test_data, use_container_width=True)
                else:
                    st.caption("No test data loaded")

elif st.session_state.tab_choice == "Data Quality":
    with data_quality_tab:
        _data_quality_tab_content()

elif st.session_state.tab_choice == "Training":
    with training_tab:
        _training_tab_content()

elif st.session_state.tab_choice == "Inference":
    with inference_tab:
        inference_tab.text_area(
            "Text for classification"
            if not is_ner()
            else "Text for entity recognition",
            key="single_text",
        )
        if inference_tab.button("Predict", key="predict_one"):
            if st.session_state.single_text:
                pred_df = predict_texts([st.session_state.single_text])
                if pred_df is not None:
                    inference_tab.dataframe(pred_df, use_container_width=True)
            else:
                inference_tab.error("Please enter text")

        inference_tab.info(
            f"CSV file should contain column: text. Max {MAX_INFERENCE_BATCH_SIZE} rows per file.",
            icon="ℹ️",
        )
        inference_tab.file_uploader(
            "Batch data",
            type="csv",
            key="inference_file",
            on_change=lambda: st.session_state.update(batch_predictions=None),
        )
        if inference_tab.button("Predict batch", key="predict_batch"):
            if st.session_state.inference_file is not None:
                df = pd.read_csv(st.session_state.inference_file)
                if "text" not in df.columns:
                    inference_tab.error("wrong columns, expected: text")
                elif len(df) > MAX_INFERENCE_BATCH_SIZE:
                    inference_tab.error(
                        f"Batch size {len(df)} exceeds maximum allowed ({MAX_INFERENCE_BATCH_SIZE}). "
                        "Use a smaller file or split the data."
                    )
                else:
                    with st.spinner("Running batch prediction..."):
                        st.session_state.batch_predictions = predict_texts(
                            df["text"].tolist()
                        )
            else:
                inference_tab.error("Please upload a file first")

        if st.session_state.batch_predictions is not None:
            inference_tab.dataframe(
                st.session_state.batch_predictions, use_container_width=True
            )

elif st.session_state.tab_choice == "Clear Cache":
    with cache_tab:
        cache_tab.button("Clear Cache", key="clear_cache", on_click=clear_cache)
