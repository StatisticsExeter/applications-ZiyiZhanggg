import plotly.express as px
import pandas as pd
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'regression'


def _boxplot(df, x_var, y_var, title):
    fig = px.box(df, x=x_var, y=y_var, title=title)
    return fig


def boxplot_age():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_energy.csv')
    fig = _boxplot(df, 'age', 'shortfall', 'Shortfall by Age Category')
    fig.write_html(VIGNETTE_DIR / 'boxplot_age.html')


def boxplot_rooms():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_energy.csv')
    fig = _boxplot(df, 'n_rooms', 'shortfall', 'Shortfall by Number of rooms')
    fig.write_html(VIGNETTE_DIR / 'boxplot_rooms.html')
