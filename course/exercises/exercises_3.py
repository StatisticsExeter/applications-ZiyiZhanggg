import plotly.express as px


def box_plot(df, cat_var, cont_var):
    """Given
      - data frame df,
      - a string 'cat_var' denoting a categorical variable and
      - a string 'continuous_var' denoting a continuous variable as well as
    return a box plot as a plotly express object
    which summarises the distribution of the continuous variable for
    different levels of cat_var."""
    return 0


def scatterplot(df, xvar, yvar):
    """Given
      - data frame df,
      - a string 'xvar' denoting a continuous variable and
      - a string 'yvar' denoting a continuous variable
    return a scatterplot plot as a plotly express object
    of the x variable against the y variable."""
    return 0


def scatterplot_groups(df, xvar, yvar, groups):
    """Given
      - data frame df,
      - a string 'xvar' denoting a continuous variable and
      - a string 'yvar' denoting a continuous variable and
      - a string 'groups' denoting a categorical variable
    return a scatterplot plot as a plotly express object
    of the x variable against the y variable that has
    markers colours for different levels of the grouping variable."""
    return 0


def scatterplot_matrix(df, numeric_cols):
    """Given
      - data frame df,
      - a list 'numeric_cols' denoting several of the continuous variables
    return a scatterplot plotmatrix as a plotly express object
    plotting each continuous variable against the others."""
    return 0


def bar_chart_means(df, cat_var, continuous_var, labels):
    """Given
      - data frame df,
      - a string 'cat_var' denoting a categorical variable and
      - a string 'continuous_var' denoting a continuous variable as well as
      - a dictionary 'labels' containing a description for the axis
        of the contents of cat_var and continuous_var
    return a bar chart as a plotly express object
    which summarises the mean of the continuous variable by different levels
    of cat_var and labels the axes using the labels dict."""
    return 0


def stacked_bar_counts(df, cat_var_1, cat_var_2, labels):
    """Given
      - data frame df,
      - a string 'cat_var_1' denoting a categorical variable for the x axis
      - a string 'cat_var_2' denoting another categorical variable for the legend
      - a dictionary 'labels' containing a description for the axis
        of the contents of cat_var_1 and cat_var_2
    return a bar chart as a plotly express object
    which summarises the mean of the continuous variable by different levels
    of cat_var and labels the axes using the labels dict."""
    return 0
