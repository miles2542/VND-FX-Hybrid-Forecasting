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

# Relative path for local library (avoiding bracket issues in absolute paths)
lib_dir <- "R_libs"
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
