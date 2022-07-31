Args <- commandArgs()
library("IsingSampler")
library("IsingFit")
library("NetworkComparisonTest")
library("bootnet")
library("qgraph")

base_type <- Args[6]
gamma <- 1
do_nct <- 1
draw_pic <- 1
base_path <- './'
y2_before <- read.csv(paste(
    base_path, '_2y_before.csv', sep = base_type
))
y2_after <- read.csv(paste(
    base_path, '_2y_after.csv', sep = base_type
))

r1 = estimateNetwork(y2_before, default = "IsingFit", directed = FALSE, tuning=gamma)
r2 = estimateNetwork(y2_after, default = "IsingFit", directed = FALSE, tuning=gamma)
if (draw_pic == 1) {
    metrics1 = centrality_auto(r1$graph, weighted = TRUE, signed = TRUE)$node.centrality
    metrics1$window = '-2_0y'
    metrics1 = data.frame(Symptom=rownames(metrics1), metrics1)
    metrics2 = centrality_auto(r2$graph, weighted = TRUE, signed = TRUE)$node.centrality
    metrics2$window = '0_2y'
    metrics2 = data.frame(Symptom=rownames(metrics2), metrics2)
    metrics = rbind(metrics1, metrics2)
    # write.csv(metrics, paste("network_cross_2025_", ".csv", sep = base_type), row.names = F)
    write.csv(metrics, paste("network_cross_", ".csv", sep = base_type), row.names = F)
    weight_trend <- paste(sum(r1$graph), sum(r2$graph), sep = ' ~ ')
    writeLines(weight_trend)
    p <- centralityPlot(r1$graph, weighted = TRUE, signed = TRUE, include = c('Strength', 'Closeness', 'Betweenness'))
    pdf(paste(
        'central_cross_', '_2y_before.pdf', sep = base_type
    ), width = 6, height = 4)
    p
    dev.off()
    p <- centralityPlot(r2$graph, weighted = TRUE, signed = TRUE, include = c('Strength', 'Closeness', 'Betweenness'))
    pdf(paste(
        'central_cross_', '_2y_after.pdf', sep = base_type
    ), width = 6, height = 4)
    p
    dev.off()
    qgraph(r1$graph,fade = FALSE, width=6, height=6)
    qgraph(r1$graph,fade = FALSE, filetype='pdf', width=6, height=6,
           labels=colnames(r1$graph), label.scale=FALSE, node.label.offset=c(0.5, -1.5), node.width=0.5,
           filename=paste(
                'network_cross_', '_2y_before.pdf', sep = base_type
            ))
    qgraph(r2$graph,fade = FALSE, width=6, height=6)
    qgraph(r2$graph,fade = FALSE, filetype='pdf', width=6, height=6,
           labels=colnames(r1$graph), label.scale=FALSE, node.label.offset=c(0.5, -1.5), node.width=0.5,
           filename=paste(
                'network_cross_', '_2y_after.pdf', sep = base_type
            ))
}
if (do_nct == 1) {
    Res_1 <- NCT(r1, r2, gamma = gamma, it=1000, binary.data = TRUE,
                 paired = TRUE, abs = FALSE,
         test.centrality = TRUE, verbose = FALSE,
                 centrality = c("closeness", "betweenness", "strength"),
         test.edges = TRUE, edges="all")
    nct_summary1 <- c(
    "-2_0y", "0_2y", Res_1$glstrinv.sep, Res_1$glstrinv.pval, Res_1$nwinv.pval
    )
    nct_node_strength <- Res_1$diffcen.pval

    nct_summary <- data.frame(rbind(nct_summary1), row.names = NULL)
    names(nct_summary) <- c('data1', 'data2', 'glstrinv_0', 'glstrinv_1', 'glstrinv.pval', 'nwinv.pval')
    write.csv(nct_summary, paste("nct_cross_", ".csv", sep = base_type), row.names = F)

    nct_node_strength <- data.frame(
      nct_node_strength
    )
    write.csv(nct_node_strength, paste("nct_cross_nodes_", ".csv", sep = base_type))
}