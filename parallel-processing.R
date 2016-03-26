library(parallel)
workerFunc <- function(n) { set.seed(n); mean(rnorm(n)) }
cluster <- makeCluster(detectCores())
values <- 10:20
system.time(result <- parLapply(cluster, values, workerFunc))
stopCluster(cluster)

library("foreach")
library("doSNOW")

cl <- makeCluster(2)
registerDoSNOW(cl)

foreach(val = values) %dopar% {
    workerFunc(val)
}
stopCluster(cl)
