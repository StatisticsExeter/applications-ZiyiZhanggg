from pathlib import Path
import pandas as pd


from course.intro.pipeline_functions import (
  tyler_viglen, calculate_correlation, filter_data,
  fit_regression, plot_scatter)


def task_check_cache():
    def check_cache():
        """Check cache folder exists"""
        models_path = Path("course/intro/cache/")
        models_path.mkdir(parents=True, exist_ok=True)
    return {
      'actions': [check_cache]
    }


def task_generate_data():
    def generate_data():
        df = tyler_viglen()
        df.to_csv('course/intro/cache/data.csv', index=False)
    return {
        'actions': [generate_data],
        'targets': ['course/intro/cache/data.csv'],
    }


def task_filter():
    def filter():
        df = pd.read_csv('course/intro/cache/data.csv')
        df = filter_data(df, 2015)
        df.to_csv('course/intro/cache/filtered_data.csv', index=False)
    return {
        'file_dep': ['course/intro/cache/data.csv'],
        'actions': [filter],
        'targets': ['course/intro/cache/filtered_data.csv'],
    }


def task_correlation():
    def correlation():
        df = pd.read_csv('course/intro/cache/filtered_data.csv')
        corr_coef, p_sig_corr = calculate_correlation(df, 'Kerosene', 'DivorceRate')
        with open('course/intro/cache/correlation.txt', 'w') as f:
            f.write(f"Correlation coefficient: {corr_coef}\\nP-value: {p_sig_corr}\\n")
    return {
        'file_dep': ['course/intro/cache/filtered_data.csv'],
        'actions': [correlation],
        'targets': ['course/intro/cache/correlation.txt'],
    }


def task_regression():
    def regression():
        df = pd.read_csv('course/intro/cache/filtered_data.csv')
        model = fit_regression(df, 'Kerosene', 'DivorceRate')
        with open('course/intro/cache/regression_summary.txt', 'w') as f:
            f.write(model.summary().as_text())
    return {
        'file_dep': ['course/intro/cache/filtered_data.csv'],
        'actions': [regression],
        'targets': ['course/intro/cache/regression_summary.txt'],
    }


def task_scatterplot():
    def scatterplot():
        df = pd.read_csv('course/intro/cache/filtered_data.csv')
        chart = plot_scatter(df, 'Kerosene', 'DivorceRate')
        chart.write_html("course/intro/cache/scatterplot.html")
    return {
        'file_dep': ['course/intro/cache/filtered_data.csv'],
        'actions': [scatterplot],
        'targets': ['course/intro/cache/scatterplot.html']
    }
