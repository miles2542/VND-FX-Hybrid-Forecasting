# R Package Dependency Manager
# Checks for existence and installs missing packages from CRAN

required_packages <- c(
  "forecast",    # ARIMA, dm.test
  "vars",        # VAR models
  "strucchange", # Bai-Perron structural breaks
  "jsonlite",    # JSON I/O
  "readr",       # Fast CSV reading
  "urca",        # Johansen co-integration
  "tseries",     # ADF and other series tests
  "lmtest",      # Breusch-Godfrey for residuals
  "yaml"         # Reading config files
)

# Resolve project root robustly across environments:
# 1) VND_FX_PROJECT_ROOT env var override
# 2) Script file path when run via Rscript/source
# 3) Current working directory fallback
resolve_project_root <- function() {
  env_root <- Sys.getenv("VND_FX_PROJECT_ROOT", unset = "")
  if (nzchar(env_root)) {
    return(normalizePath(env_root, winslash = "/", mustWork = FALSE))
  }

  script_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", script_args, value = TRUE)
  if (length(file_arg) > 0) {
    script_file <- sub("^--file=", "", file_arg[1])
    return(normalizePath(file.path(dirname(script_file), ".."), winslash = "/", mustWork = FALSE))
  }

  source_path <- NULL
  for (i in sys.nframe():1) {
    frame <- sys.frame(i)
    if (!is.null(frame$ofile) && nzchar(frame$ofile)) {
      source_path <- frame$ofile
      break
    }
  }
  if (!is.null(source_path)) {
    return(normalizePath(file.path(dirname(source_path), ".."), winslash = "/", mustWork = FALSE))
  }

  fallback <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  cat(paste0("[WARN] Could not detect script path. Using current working directory as project root: ", fallback, "\n"))
  fallback
}

project_root <- resolve_project_root()

lib_dir <- file.path(project_root, "R_libs")
if (!dir.exists(lib_dir)) dir.create(lib_dir, recursive = TRUE)
lib_path <- normalizePath(lib_dir, winslash = "/")

cat(paste0("--- Verifying R Dependencies (CRAN) ---\nLocal Lib: ", lib_path, "\n"))

# Ensure lib_path is the first place R looks
.libPaths(c(lib_path, .libPaths()))

install_if_missing <- function(pkg) {
  # Check if package is installed in OUR local path
  # This ensures the pipeline is portable and self-contained
  if (!pkg %in% rownames(installed.packages(lib.loc = lib_path))) {
    cat(paste0("[INFO] Installing package to local lib: ", pkg, "\n"))
    # Use official cloud mirror for stability
    install.packages(pkg, repos = "https://cloud.r-project.org/", lib = lib_path)
  } else {
    cat(paste0("[OK] Package exists in local lib: ", pkg, "\n"))
  }
}

invisible(lapply(required_packages, install_if_missing))

cat("--- R Dependencies Verified ---\n")
