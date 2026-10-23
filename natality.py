import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import requests
from pathlib import Path
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

"""
 libreria requests per effettuare il download dei file dal repository online
https://www.nber.org/research/data/vital-statistics-natality-birth-data
i dati sono disponibili sia in formato csv chein formato Stata (.dta) e possono essere letti con pandas.read_stata
sono disponibili i dati dal 1994 al 2024, ma i dati del 2024 sono ancora incompleti e non contengono tutte le variabili
in una prima fase scarico solo uno o due anni, ma poi per un'addestramento più robusto conviene scaricare tutti gli anni disponibili e unirli in un unico dataset
"""

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR / "data"
BASE_URL = "https://data.nber.org/nvss/natality/dta"
YEARS = [2023]


def get_dataset_path(year, dataset_type):
	DATA_DIR.mkdir(exist_ok=True)
	filename = f"natality{year}{dataset_type}.dta"
	data_path = DATA_DIR / filename
	legacy_path = BASE_DIR / filename

	if data_path.exists():
		print(f"File già presente: {data_path}")
		return data_path
	if legacy_path.exists():
		print(f"Uso il file già presente: {legacy_path}")
		return legacy_path

	url = f"{BASE_URL}/{year}/{filename}"
	temp_path = data_path.with_suffix(data_path.suffix + ".part")
	print(f"Scarico {filename}...")
	with requests.get(url, stream=True, timeout=(10, 120)) as response:
		response.raise_for_status()
		with temp_path.open("wb") as output:
			for chunk in response.iter_content(chunk_size=8*1024 * 1024):
				if chunk:
					output.write(chunk)
	temp_path.replace(data_path)
	return data_path


if not YEARS:
	raise ValueError("Selezionare almeno un anno in YEARS")
if any(year < 1994 or year > 2024 for year in YEARS):
	raise ValueError("Sono supportati gli anni dal 1994 al 2024")


def read_stata_with_labels(path, fraction=0.2):
	reader = pd.read_stata(path, iterator=True)
	reader._ensure_open()
	rows_to_read = max(1, int(reader._nobs * fraction))
	data = reader.read(rows_to_read)
	labels = {
		"variable_labels": reader.variable_labels(),
		"value_labels": reader.value_labels(),
	}
	return data, labels


datasets = []
dataset_labels = {}
for year in YEARS:
	path_us = get_dataset_path(year, "us")
	path_ps = get_dataset_path(year, "ps")

	nat_us, dataset_labels[(year, "us")] = read_stata_with_labels(path_us)
	nat_ps, dataset_labels[(year, "ps")] = read_stata_with_labels(path_ps)
	nat_year = pd.concat([nat_us, nat_ps], ignore_index=True, sort=False)
#	nat_year = nat_ps.copy()
	year_column = pd.DataFrame({"natality_year": np.full(len(nat_year), year)})
	nat_year = pd.concat([nat_year, year_column], axis=1)
	datasets.append(nat_year)

nat = pd.concat(datasets, ignore_index=True, sort=False)

print(nat.head())
print(f"Righe unite: {len(nat)}")
first_year_us_labels = dataset_labels[(YEARS[0], "ps")]
variable_labels_table = pd.DataFrame(
	first_year_us_labels["variable_labels"].items(),
	columns=["Variabile", "Label"],
)
print(f"\nLabel delle variabili {YEARS[0]} ps:")
print(variable_labels_table.to_string(index=False))
labels_csv_path = DATA_DIR / f"variable_labels_{YEARS[0]}_ps.csv"
variable_labels_table.to_csv(labels_csv_path, index=False)
print(f"\nLabel salvate in: {labels_csv_path}")

target_groups = {
	"target_morbilita_materna": ["mm_mtr", "mm_plac", "mm_rupt", "mm_uhyst", "mm_aicu"],
	"target_complicazioni_neonatali": ["ab_aven1", "ab_aven6", "ab_nicu", "ab_surf", "ab_anti", "ab_seiz"],
}

candidate_columns = nat.columns[:118].tolist()
variable_labels = {
	column: label
	for labels in dataset_labels.values()
	for column, label in labels["variable_labels"].items()
}
target_source_columns = {
	column for columns in target_groups.values() for column in columns
}
reporting_flag_columns = [
	column for column in candidate_columns
	if "reporting flag" in variable_labels.get(column, "").lower()
]
input_columns = [
	column for column in candidate_columns
	if column not in reporting_flag_columns
	and column not in target_source_columns
	and column != "natality_year"
	and column != "no_risks"
]
print(f"Feature selezionate: {len(input_columns)} su {len(candidate_columns)} candidate")
print(f"Reporting flag escluse: {reporting_flag_columns}")
print(f"Variabili target escluse dalle feature: {sorted(target_source_columns & set(candidate_columns))}")
print(f"Variabili riassuntive escluse: {sorted(set(candidate_columns) - set(input_columns) - set(reporting_flag_columns) - target_source_columns - {'natality_year'})}")


def to_binary_event(series):
	labels = series.astype("string").str.strip().str.lower()
	positive = labels.str.contains(r"\b(?:yes|true|confirmed)\b", na=False)
	positive |= labels.str.match(r"^[yc]\b", na=False)
	negative = labels.str.contains(r"\b(no|false)\b", na=False)
	negative |= labels.str.match(r"^n\b", na=False)
	result = pd.Series(np.nan, index=series.index, dtype="float64")
	result.loc[positive] = 1
	result.loc[negative] = 0
	return result


target_data = pd.DataFrame(index=nat.index)
for target_name, source_columns in target_groups.items():
	missing_columns = [column for column in source_columns if column not in nat.columns]
	if missing_columns:
		raise ValueError(f"Variabili necessarie per {target_name} non trovate: {missing_columns}")
	events = pd.DataFrame(
		{column: to_binary_event(nat[column]) for column in source_columns},
		index=nat.index,
	)
	any_positive = events.eq(1).any(axis=1)
	all_negative = events.eq(0).all(axis=1)
	target_data[target_name] = np.select(
		[any_positive, all_negative], [1.0, 0.0], default=np.nan
	)

numeric_columns = nat[input_columns].select_dtypes(include="number").columns.tolist()
categorical_columns = [column for column in input_columns if column not in numeric_columns]
preprocessor = ColumnTransformer(
	[
		(
			"numeric",
			Pipeline([
				("imputer", SimpleImputer(strategy="median")),
			]),
			numeric_columns,
		),
		(
			"categorical",
			Pipeline([
				("imputer", SimpleImputer(strategy="most_frequent")),
				("onehot", OneHotEncoder(handle_unknown="ignore")),
			]),
			categorical_columns,
		),
	]
)

models = {}
for target_name in target_data.columns:
	known_target = target_data[target_name].notna()
	X = nat.loc[known_target, input_columns]
	y = target_data.loc[known_target, target_name].astype("int8")
	if y.nunique() < 2:
		print(f"\n{target_name}: regressione non eseguita, presente una sola classe.")
		continue
	if y.value_counts().min() < 10:
		print(f"\n{target_name}: meno di 10 casi nella classe minoritaria, split stratificati non affidabili.")
		continue
	print(f"\nCross-validation per {target_name} (righe utilizzate: {len(y)}):")
	comparison_rows = []
	fold_results = {
		(class_weight, threshold_name): []
		for class_weight in ("nessuno", "bilanciato")
		for threshold_name in ("default 0.50", "soglia da validazione")
	}
	importance_totals = {class_weight: {} for class_weight in ("nessuno", "bilanciato")}
	importance_counts = {class_weight: {} for class_weight in ("nessuno", "bilanciato")}
	models_by_weight = {class_weight: [] for class_weight in ("nessuno", "bilanciato")}
	cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	for fold, (train_indices, test_indices) in enumerate(cross_validator.split(X, y), start=1):
		X_fold_train = X.iloc[train_indices]
		X_fold_test = X.iloc[test_indices]
		y_fold_train = y.iloc[train_indices]
		y_fold_test = y.iloc[test_indices]
		X_train, X_valid, y_train, y_valid = train_test_split(
			X_fold_train, y_fold_train, test_size=0.2, random_state=42 + fold,
			stratify=y_fold_train,
		)
		positive_count = int((y_train == 1).sum())
		negative_count = int((y_train == 0).sum())
		class_weight_options = (
			("nessuno", 1.0),
			("bilanciato", negative_count / positive_count),
		)
		for class_weight, positive_weight in class_weight_options:
			fold_preprocessor = clone(preprocessor)
			X_train_processed = fold_preprocessor.fit_transform(X_train, y_train)
			X_valid_processed = fold_preprocessor.transform(X_valid)
			X_fold_test_processed = fold_preprocessor.transform(X_fold_test)
			xgboost_model = XGBClassifier(
				n_estimators=2000,
				max_depth=6,
				learning_rate=0.05,
				subsample=0.8,
				colsample_bytree=0.8,
				scale_pos_weight=positive_weight,
				eval_metric="aucpr",
				early_stopping_rounds=50,
				tree_method="hist",
				n_jobs=-1,
				 random_state=42,
			)
			xgboost_model.fit(
				X_train_processed,
				y_train,
				eval_set=[(X_valid_processed, y_valid)],
				verbose=False,
			)
			model = Pipeline([
				("preprocessor", fold_preprocessor),
				("xgboost", xgboost_model),
			])
			models_by_weight[class_weight].append(model)
			print(
				f"    {class_weight}: early stop al round "
				f"{xgboost_model.best_iteration + 1}"
			)

			feature_names = fold_preprocessor.get_feature_names_out()
			feature_importances = xgboost_model.feature_importances_
			for feature, importance in zip(feature_names, feature_importances):
				importance_totals[class_weight][feature] = (
					importance_totals[class_weight].get(feature, 0.0) + importance
				)
				importance_counts[class_weight][feature] = (
					importance_counts[class_weight].get(feature, 0) + 1
				)

			valid_probabilities = xgboost_model.predict_proba(X_valid_processed)[:, 1]
			precision_values, recall_values, thresholds = precision_recall_curve(
				y_valid, valid_probabilities
			)
			validation_f1 = (
				2 * precision_values[:-1] * recall_values[:-1]
				/ (precision_values[:-1] + recall_values[:-1] + 1e-12)
			)
			best_threshold = thresholds[int(np.argmax(validation_f1))]
			test_probabilities = xgboost_model.predict_proba(X_fold_test_processed)[:, 1]
			for threshold_name, threshold in (
				("default 0.50", 0.5),
				("soglia da validazione", best_threshold),
			):
				test_predictions = (test_probabilities >= threshold).astype("int8")
				fold_results[(class_weight, threshold_name)].append({
					"threshold": threshold,
					"precision": precision_score(y_fold_test, test_predictions, zero_division=0),
					"recall": recall_score(y_fold_test, test_predictions, zero_division=0),
					"f1": f1_score(y_fold_test, test_predictions, zero_division=0),
				})
		print(f"  Fold {fold}/5 completato")

	for class_weight in ("nessuno", "bilanciato"):
		models[(target_name, class_weight)] = models_by_weight[class_weight]
		average_importance = {
			feature: importance_totals[class_weight][feature] / importance_counts[class_weight][feature]
			for feature in importance_totals[class_weight]
		}
		top_features = pd.Series(average_importance).sort_values(ascending=False).head(15)
		print(f"\nFeature più importanti medie ({class_weight}):")
		print(top_features.rename_axis("feature").reset_index(name="importance").to_string(
			index=False, float_format=lambda value: f"{value:.4f}"
		))
		for threshold_name in ("default 0.50", "soglia da validazione"):
			fold_metrics = fold_results[(class_weight, threshold_name)]
			comparison_rows.append({
				"modello": "XGBoost",
				"pesi": class_weight,
				"soglia": threshold_name,
				"valore soglia media": np.mean([result["threshold"] for result in fold_metrics]),
				"precision media": np.mean([result["precision"] for result in fold_metrics]),
				"precision std": np.std([result["precision"] for result in fold_metrics]),
				"recall media": np.mean([result["recall"] for result in fold_metrics]),
				"recall std": np.std([result["recall"] for result in fold_metrics]),
				"F1 media": np.mean([result["f1"] for result in fold_metrics]),
				"F1 std": np.std([result["f1"] for result in fold_metrics]),
			})
	print(pd.DataFrame(comparison_rows).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
