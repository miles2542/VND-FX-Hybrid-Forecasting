lib_path <- "x:/Programming/Python/[Y3S2] Year 3, Spring semester/[Y3S2] Time Series Analysis/Final project/V5 - Ince & Traflis (2006)/R_libs"
cat(paste0("Testing path: ", lib_path, "\nExists: ", dir.exists(lib_path), "\n"))
.libPaths(c(lib_path, .libPaths()))
print(.libPaths())
library(tseries)
cat("SUCCESS: Loaded tseries\n")
