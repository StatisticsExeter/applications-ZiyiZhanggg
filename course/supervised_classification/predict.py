import joblib
import pandas as pd
from course.utils import find_project_root


def predict(model_path, X_test_path, y_pred_path, y_pred_prob_path):
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)

    y_pred = model.predict(X_test)
    pd.DataFrame({"predicted_built_age": y_pred}).to_csv(y_pred_path, index=False)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            y_score = proba[:, 1]
        else:
            y_score = proba.ravel()
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test).ravel()
    else:
        y_score = pd.Series(y_pred).astype("category").cat.codes.values

    pd.DataFrame({"predicted_built_age": y_score}).to_csv(y_pred_prob_path, index=False)

    y_pred_series = pd.Series(y_pred, name='predicted_built_age')
    y_pred_series.to_csv(y_pred_path, index=False)


def pred_lda():
    base_dir = find_project_root()
    model_path = base_dir / 'data_cache' / 'models' / 'lda_model.joblib'
    X_test_path = base_dir / 'data_cache' / 'energy_X_test.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'lda_y_pred.csv'
    y_pred_prob_path = base_dir / 'data_cache' / 'models' / 'lda_y_pred_prob.csv'
    predict(model_path, X_test_path, y_pred_path, y_pred_prob_path)


def pred_qda():
    base_dir = find_project_root()
    model_path = base_dir / 'data_cache' / 'models' / 'qda_model.joblib'
    X_test_path = base_dir / 'data_cache' / 'energy_X_test.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'qda_y_pred.csv'
    y_pred_prob_path = base_dir / 'data_cache' / 'models' / 'qda_y_pred_prob.csv'
    predict(model_path, X_test_path, y_pred_path, y_pred_prob_path)
