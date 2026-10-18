options(repos = c(CRAN = "https://cloud.r-project.org"))
pkgs <- c("qqman", "data.table")
for (p in pkgs) if (!requireNamespace(p, quietly=TRUE)) install.packages(p, Ncpus=4)
