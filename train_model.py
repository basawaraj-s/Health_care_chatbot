"""Utility script to train the healthcare chatbot classifier."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


DATA_DIR = Path("data")
DATASET_PATH = DATA_DIR / "Diseases_and_Symptoms_dataset.csv"
TRAIN_OUTPUT = DATA_DIR / "training.csv"
TEST_OUTPUT = DATA_DIR / "testing.csv"
SYMPTOMS_OUTPUT = DATA_DIR / "symptoms.csv"

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "random_forest_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

N_ESTIMATORS = 160
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 1

ENABLE_HYPERPARAM_TUNING = os.environ.get("SKIP_HYPERPARAM_TUNING", "0") != "1"
# Optional quick mode to ensure fast training during local runs/CI.
# When QUICK_TRAIN=1, we reduce model size and depth so the script finishes quickly.
QUICK_TRAIN = os.environ.get("QUICK_TRAIN", "0") == "1"
HYPERPARAM_SEARCH_ITERATIONS = 18
CV_FOLDS = 3

HYPERPARAM_DISTRIBUTIONS = {
	"n_estimators": [240, 320, 400, 480, 560],
	"max_depth": [None, 32, 48, 64],
	"min_samples_split": [2, 4, 6, 8, 10],
	"min_samples_leaf": [1, 2, 3, 4],
	"max_features": ["sqrt", "log2", 0.35, 0.45, 0.6],
	"criterion": ["gini", "entropy", "log_loss"],
	"class_weight": ["balanced", "balanced_subsample"],
}


def load_dataset(path: Path) -> Tuple[pd.DataFrame, List[str]]:
	"""Load and clean the raw dataset."""

	if not path.exists():
		raise FileNotFoundError(f"Dataset not found at {path}")

	df = pd.read_csv(path)
	if df.empty:
		raise ValueError("Dataset is empty")

	first_col = df.columns[0]
	df = df.rename(columns={first_col: "disease"})

	df = df.dropna(subset=["disease"]).copy()
	df["disease"] = df["disease"].astype(str).str.strip()

	feature_cols = [c for c in df.columns if c != "disease"]
	df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

	df = df.drop_duplicates().reset_index(drop=True)

	# Optional: downsample per class for quick local iterations
	if QUICK_TRAIN:
		max_per_class = 120
		before = len(df)
		df = (
			df.groupby("disease", group_keys=False)
			.apply(lambda g: g.sample(n=min(max_per_class, len(g)), random_state=42))
			.reset_index(drop=True)
		)
		after = len(df)
		print(f"[quick-train] Downsampled dataset per class: {before} -> {after} rows (<= {max_per_class} per class)")

	return df, feature_cols


def build_display_map(feature_names: List[str]) -> Dict[str, str]:
	"""Create a user-friendly label for each symptom feature."""

	display_map: Dict[str, str] = {}
	collisions: Dict[str, int] = {}

	for name in feature_names:
		cleaned = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
		cleaned = re.sub(r"\s+", " ", cleaned)
		display_label = cleaned.title() if cleaned else name

		count = collisions.get(display_label, 0)
		if count:
			collisions[display_label] = count + 1
			display_key = f"{display_label} ({count + 1})"
		else:
			collisions[display_label] = 1
			display_key = display_label

		display_map[display_key] = name

	return display_map


def prepare_splits(df: pd.DataFrame, feature_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Split the dataset into training and testing frames."""

	train_df, test_df = train_test_split(
		df,
		test_size=0.2,
		random_state=42,
		stratify=df["disease"],
	)

	train_df = train_df.sort_values("disease").reset_index(drop=True)
	test_df = test_df.sort_values("disease").reset_index(drop=True)

	return train_df, test_df



def tune_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[RandomForestClassifier, Dict[str, object]]:
	"""Optionally run randomized search to tune the random forest."""

	# Adjust base params if QUICK_TRAIN is enabled for faster iteration
	base_params = {
		"n_estimators": 64 if QUICK_TRAIN else N_ESTIMATORS,
		"max_depth": 20 if QUICK_TRAIN else MAX_DEPTH,
		"min_samples_leaf": 2 if QUICK_TRAIN else MIN_SAMPLES_LEAF,
		"random_state": 42,
		"n_jobs": -1,
		"class_weight": "balanced_subsample",
	}
	model = RandomForestClassifier(**base_params)
	if not ENABLE_HYPERPARAM_TUNING:
		model.fit(X_train, y_train)
		return model, {
			"tuning_enabled": False,
			"base_params": {key: value for key, value in base_params.items() if key != "random_state"},
		}

	print("Running randomized hyperparameter search...")
	if QUICK_TRAIN:
		print("[quick-train] Note: QUICK_TRAIN=1 is set; hyperparameter search may still be slow.")
	search = RandomizedSearchCV(
		estimator=model,
		param_distributions=HYPERPARAM_DISTRIBUTIONS,
		n_iter=HYPERPARAM_SEARCH_ITERATIONS,
		scoring="f1_weighted",
		n_jobs=-1,
		cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42),
		verbose=1,
		random_state=42,
		refit=True,
	)
	search.fit(X_train, y_train)
	best_model: RandomForestClassifier = search.best_estimator_
	details = {
		"tuning_enabled": True,
		"best_params": search.best_params_,
		"best_cv_score": float(search.best_score_),
		"cv_folds": CV_FOLDS,
		"n_iter": HYPERPARAM_SEARCH_ITERATIONS,
		"scoring": "f1_weighted",
	}
	return best_model, details


def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_names: List[str]) -> Tuple[RandomForestClassifier, LabelEncoder, Dict[str, float], Dict[str, object]]:
	"""Train a random forest classifier and report metrics."""

	X_train = train_df[feature_names].values
	X_test = test_df[feature_names].values

	encoder = LabelEncoder()
	y_train = encoder.fit_transform(train_df["disease"].values)
	y_test = encoder.transform(test_df["disease"].values)

	print(
		"Starting training...",
		f"samples={X_train.shape[0]}",
		f"features={X_train.shape[1]}",
		f"estimators={N_ESTIMATORS}",
	)

	clf, tuning_details = tune_random_forest(X_train, y_train)

	y_pred = clf.predict(X_test)

	accuracy = float(accuracy_score(y_test, y_pred))
	f1 = float(f1_score(y_test, y_pred, average="weighted"))
	report = classification_report(
		y_test,
		y_pred,
		target_names=encoder.classes_,
		output_dict=True,
		zero_division=0,
	)
	report_serializable = {
		k: {metric: float(val) for metric, val in metrics.items()}
		if isinstance(metrics, dict)
		else float(metrics)
		for k, metrics in report.items()
	}

	metrics = {
		"accuracy": accuracy,
		"f1_weighted": f1,
		"classification_report": report_serializable,
	}
	metrics["tuning"] = tuning_details

	model_params = {
		key: value
		for key, value in clf.get_params().items()
		if isinstance(value, (int, float, str, bool)) or value is None
	}
	feature_importance = None
	if hasattr(clf, "feature_importances_"):
		importances = clf.feature_importances_
		indices = np.argsort(importances)[::-1][:25]
		feature_importance = [
			{"feature": feature_names[idx], "importance": float(importances[idx])}
			for idx in indices
		]

	training_summary: Dict[str, object] = {
		"hyperparameters": model_params,
		"top_feature_importances": feature_importance,
	}

	print(f"Accuracy: {accuracy:.4f}")
	print(f"Weighted F1: {f1:.4f}")
	print(classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0))

	return clf, encoder, metrics, training_summary


def persist_artifacts(
	classifier: RandomForestClassifier,
	encoder: LabelEncoder,
	feature_names: List[str],
	display_map: Dict[str, str],
	metrics: Dict[str, float],
	training_summary: Dict[str, object],
) -> None:
	"""Store the trained model and metadata bundle."""

	ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

	payload = {
		"model": classifier,
		"label_encoder": encoder,
	}
	joblib.dump(payload, MODEL_PATH)

	metadata = {
		"feature_names": feature_names,
		"symptom_display_map": display_map,
		"classes": encoder.classes_.tolist(),
		"metrics": metrics,
		"training_summary": training_summary,
	}
	with open(METADATA_PATH, "w", encoding="utf-8") as handle:
		json.dump(metadata, handle, indent=2)

	print(f"Saved model to {MODEL_PATH}")
	print(f"Saved metadata to {METADATA_PATH}")


def persist_support_files(
	train_df: pd.DataFrame,
	test_df: pd.DataFrame,
	display_map: Dict[str, str],
) -> None:
	"""Write helpful CSV files referenced by the Streamlit UI."""

	TRAIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

	train_df.to_csv(TRAIN_OUTPUT, index=False)
	test_df.to_csv(TEST_OUTPUT, index=False)

	symptoms_records = [
		{"symptom": display, "feature_code": code}
		for display, code in sorted(display_map.items(), key=lambda item: item[0])
	]
	pd.DataFrame(symptoms_records).to_csv(SYMPTOMS_OUTPUT, index=False)

	print(f"Wrote training split to {TRAIN_OUTPUT}")
	print(f"Wrote testing split to {TEST_OUTPUT}")
	print(f"Wrote symptom catalogue to {SYMPTOMS_OUTPUT}")


def main() -> None:
	df, feature_names = load_dataset(DATASET_PATH)
	display_map = build_display_map(feature_names)

	train_df, test_df = prepare_splits(df, feature_names)

	classifier, encoder, metrics, training_summary = train_model(train_df, test_df, feature_names)

	persist_artifacts(classifier, encoder, feature_names, display_map, metrics, training_summary)
	persist_support_files(train_df, test_df, display_map)


if __name__ == "__main__":
	main()
 