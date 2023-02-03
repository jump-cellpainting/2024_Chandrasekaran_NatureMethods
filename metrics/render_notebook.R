render_notebook <-
  function(input, output_dir = ".", ...) {
    output_file <- paste0(tools::file_path_sans_ext(input),
                          ".md")
    
    rmarkdown::render(
      input,
      output_file = output_file,
      output_dir = output_dir,
      output_format = "github_document",
      ...
    )
    
    output_file_rel <- file.path(output_dir, output_file)
    
    strip_text <- paste0(normalizePath(path.expand(output_dir)), "/")
    
    logger::log_debug(glue::glue("Rendered: input={input}"))
    
    logger::log_debug(glue::glue("Rendered: strip_text={strip_text}"))
    
    read_lines(output_file_rel) %>%
      str_remove_all(strip_text) %>%
      write_lines(output_file_rel)
    
  }
