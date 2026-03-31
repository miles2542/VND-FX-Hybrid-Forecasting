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
  "lmtest"       # Breusch-Godfrey for residuals
)

cat("--- Verifying R Dependencies (CRAN) ---\n")

install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    cat(paste0("[INFO] Installing package: ", pkg, "\n"))
    install.packages(pkg, repos = "https://cran.microsoft.com/snapshot/2024-03-31/")
  } else {
    cat(paste0("[OK] Package already exists: ", pkg, "\n"))
  }
}

invisible(lapply(required_packages, install_if_missing))

cat("--- R Dependencies Verified ---\n")
