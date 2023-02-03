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
`~/work/projects/2019_07_11_JUMP-CP/workspace/software/nelisa-cellpainting/2021_Chandrasekaran_submitted/metrics`

```r
setwd("~/Downloads/evalzoo/matric")

library(tidyverse)

source("run_param.R")

results_root_dir <-
  "~/work/projects/2019_07_11_JUMP-CP/workspace/software/2021_Chandrasekaran_submitted/metrics"

params_root_dir <- results_root_dir

run_all <- function(config_list) {
  config_list %>%
    walk(function(i)
      run_param(
        param_file = file.path(params_root_dir, "params", i),
        results_root_dir = results_root_dir
      ))
}

c("params_cpjump1_prod_technical.yaml") %>% run_all()
```


Generate a TOC like this

```r
rmarkdown::render("6.results_toc.Rmd", params = list(configs = configs, results_root_dir = results_root_dir))
```