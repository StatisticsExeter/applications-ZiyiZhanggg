import pandas as pd
import importlib
import course.exercises.exercises_3 as ex3
importlib.reload(ex3)
from course.exercises.exercises_3 import (  # noqa: E402
  bar_chart_means, stacked_bar_counts, box_plot, scatterplot,
  scatterplot_groups, scatterplot_matrix)


# Load the Palmer Penguins data (have a look at these data)
# https://allisonhorst.github.io/palmerpenguins/articles/intro.html
penguins = pd.read_csv("data/penguins.csv")

# boxplot
bp = box_plot(penguins, 'species', 'flipper_length_mm')
bp.write_html("flipper_lenth_by_species.html")

# scatterplot
sp = scatterplot(penguins, "bill_length_mm", "bill_depth_mm")
sp.write_html("bill_length_versus_bill_depth.html")

# scatterplot with groups
spg = scatterplot_groups(penguins, "bill_length_mm", "bill_depth_mm", "species")
spg.write_html("bill_length_by_bill_depth_by_species.html")

# scatterplot matrix
numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
spm = scatterplot_matrix(penguins, numeric_cols)
spm.write_html("scatterplot_matrix.html")

# Functions that need both Pandas data manipulation AND plotting
# bar chart of group means
labels = {cat_var: "Penguin Species",  # noqa: F821
          continuous_var: "Average Bill Length (mm)"}  # noqa: F821
bc = bar_chart_means(penguins, 'species', 'bill_length_mm', labels)
bc.write_html("mean_bill_length_by_species.html")

# stacked bar chart of counts
labels = {cat_var: "Penguin Species"}  # noqa: F821
sbc = stacked_bar_counts(penguins, 'species', 'sex', labels)
sbc.write_html("count_of_sex_and_species.html")
