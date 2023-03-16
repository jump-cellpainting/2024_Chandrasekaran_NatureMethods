# Calculate metrics

- Run `1.setup.Rmd` to generate `profiles`
- Clone <https://github.com/cytomining/evalzoo>
- Follow instructions in <https://github.com/cytomining/evalzoo/tree/main/matric#computational-environment> for setting up the computational environment
- Run calculator using instructions in <https://github.com/cytomining/evalzoo/tree/main/matric#evaluate-metrics-using-matric>

Example:

```sh
cd ~/Downloads
git clone git@github.com:cytomining/evalzoo.git
```

Open `evalzoo.Rproj` in RStudio and then restore the R environment:

```r
renv:restore()
```

Restart R, and you now ready to run the calculator

For this example, the parent folder of this README is
`~/work/projects/2019_07_11_JUMP-CP/workspace/software/2021_Chandrasekaran_submitted/metrics`

```r
# update this to the actual path on your computer
setwd("~/Downloads/evalzoo/matric")

library(tidyverse)

source("run_param.R")

# update this to the actual path on your computer
# parent directory of the results folder
results_root_dir <-
  "~/work/projects/2019_07_11_JUMP-CP/workspace/software/2021_Chandrasekaran_submitted/metrics"

# parent directory of the params folder
params_root_dir <- results_root_dir

# function to run calculator on the list param files
run_all <- function(config_list) {
  config_list %>%
    walk(function(i)
      run_param(
        param_file = file.path(params_root_dir, "params", i),
        results_root_dir = results_root_dir
      ))
}

# run just one file
c("params_cpjump1_prod_technical.yaml") %>% run_all()
```

Generate a TOC like this

```r
configs <- list.files(file.path(results_root_dir, "results"), pattern = "[a-z0-9]{8}")
rmarkdown::render("6.results_toc.Rmd", params = list(configs = configs, results_root_dir = results_root_dir))
```
