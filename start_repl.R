library(reticulate)
ENV_NAME <- "python-exercises"
envs <- conda_list()
target <- envs[envs$name == ENV_NAME, ]

if (nrow(target) == 0) {
  stop(sprintf(
    "Conda environment '%s' not found. Please create it or activate it.",
    ENV_NAME
  ))
}

python_env <- target$python
use_python(python_env, required = TRUE)

py_config()

reticulate::repl_python()

Sys.setenv(RETICULATE_CONDA_PATH = "D:/Anaconda/condabin/conda.bat")
reticulate::py_config() 


