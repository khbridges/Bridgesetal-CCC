library(ggplot2)
library(plyr)
library(gridExtra)
library(magrittr)
library(tidyr)
library(raster)
library(OpenImageR)
library(ggpubr)
library(grid)
library(wesanderson)
library(RColorBrewer)
library(dplyr)
library(tidyverse)
library(Seurat)
library(SeuratData)
library(Matrix)
library(janitor)
library(rio)
library(readxl)
library(MERINGUE)

source('DBIT-func.R')

# set the expression matrix containing folder as the working directory
file_dir <- '/Users/katebridges/Downloads/YUMMER_ThirdBatch_Treatments'
file_dir_ <- '/Users/katebridges/Downloads/YUMMER_SecondBatch'

# ICB <- get_gene_UMI_count(file.path(file_dir, 'CPI_0610'), 'CPI_0610_repaired.csv')
# comb <- get_gene_UMI_count(file.path(file_dir, 'CPICD40_0610'), 'CPICD40_0610_repaired.csv')

# generating Seurat obj also normalizes & clusters data by default
ctrl.obj <- generate_Seurat_obj(file_dir_, 'YRctrl', 'YUM_gex_secondbatch_updated.tsv', 0.9, 'YRctrl')
icb.obj <- generate_Seurat_obj(file.path(file_dir, 'CPI_0610'), 'YRicb', 'CPI_0610_repaired.csv', 0.9, 'YRicb', sept = ',', transp = FALSE)
comb.obj <- generate_Seurat_obj(file.path(file_dir, 'CPICD40_0610'), 'YRcomb', 'CPICD40_0610_repaired.csv', 0.9, 'YRcomb', sept = ',', transp = FALSE)

# viz data by cluster (6A)
SpatialDimPlot(ctrl.obj,  group.by = "seurat_clusters", pt.size.factor = 5) + 
  scale_x_continuous(name = "X", expand = expansion(mult = c(0.008, 0.008))) +
  scale_y_continuous(name = "Y", expand = expansion(mult = c(0.008, 0.008))) +
  theme(legend.position = "right") + NoAxes()

SpatialDimPlot(icb.obj,  group.by = "seurat_clusters", pt.size.factor = 5) + 
  scale_x_continuous(name = "X", expand = expansion(mult = c(0.008, 0.008))) +
  scale_y_continuous(name = "Y", expand = expansion(mult = c(0.008, 0.008))) +
  theme(legend.position = "right") + NoAxes()

SpatialDimPlot(comb.obj,  group.by = "seurat_clusters", pt.size.factor = 5) + 
  scale_x_continuous(name = "X", expand = expansion(mult = c(0.008, 0.008))) +
  scale_y_continuous(name = "Y", expand = expansion(mult = c(0.008, 0.008))) +
  theme(legend.position = "right") + NoAxes()

# scoring data by gene signatures of interest - reading in DEGs from scRNA-seq
gene_dir <- '/Users/katebridges/Downloads/celltype-signature-genes.xlsx'
sigs <- import_list(gene_dir, header=FALSE) # this excel sheet contains top100 genes

# limit signature genes to ones that are consistently detected across datasets
sigs.detected <- c()

for (g in names(sigs)) {
  top20 <- c()
  i = 1
  while(length(top20) < 20) {
    j = sigs[[g]]$...1[i]
    if (length(which(rownames(ctrl.obj@assays$SCT) == j)) > 0 & 
        length(which(rownames(icb.obj@assays$SCT) == j)) > 0 &
        length(which(rownames(comb.obj@assays$SCT) == j)) > 0) {
      top20 <- c(top20, j)
    }
    i = i + 1
  } 
  sigs.detected[[g]] <- top20
}

# write to file
openxlsx::write.xlsx(sigs.detected, '/Users/katebridges/Downloads/20240801-sigsdetected-DBiT.xlsx')

# score each pixel in each dataset by cell type signatures using 100 random control genes
ctrl.obj <- AddModuleScore(object = ctrl.obj, features = sigs.detected, ctrl=100, assay='SCT', nbin=15)
icb.obj <- AddModuleScore(object = icb.obj, features = sigs.detected, ctrl=100, assay='SCT', nbin=15)
comb.obj <- AddModuleScore(object = comb.obj, features = sigs.detected, ctrl=100, assay='SCT', nbin=15)

# viz of cell type signature scores - macro (cluster1), treg (cluster4), cd8 (cluster3)
for (k in c('Cluster1', 'Cluster4', 'Cluster3')) {
  print(SpatialFeaturePlot(ctrl.obj,  features = k, pt.size.factor = 5, min.cutoff=0, max.cutoff=0.2) + 
          scale_x_continuous(name = "X", expand = expansion(mult = c(0.008, 0.008))) +
          scale_y_continuous(name = "Y", expand = expansion(mult = c(0.008, 0.008))) +
          theme(legend.position = "right") + NoAxes())
}

# correlation of features of interest - writing bootstrapped corr results to file
# for viz with GraphPad 

# mac-treg corr
ctrl.mac.treg <- corr_CV(ctrl.obj, 'Cluster1', 'Cluster4', 20, rep=100)
icb.mac.treg <- corr_CV(icb.obj, 'Cluster1', 'Cluster4',20, rep=100)
comb.mac.treg <- corr_CV(comb.obj, 'Cluster1', 'Cluster4', 20, rep=100)
mac.treg <- tibble('ctrl' = ctrl.mac.treg$corr, 
                   'icb' = icb.mac.treg$corr,
                   'comb' = comb.mac.treg$corr)
openxlsx::write.xlsx(mac.treg, '/Users/katebridges/Downloads/20240801_mac-Treg_corr-20x20-100.xlsx')

# mac-cd8 t cell cor
ctrl.mac.cd8 <- corr_CV(ctrl.obj, 'Cluster1', 'Cluster3', 10, rep=1000)
icb.mac.cd8 <- corr_CV(icb.obj, 'Cluster1', 'Cluster3', 10, rep=1000)
comb.mac.cd8 <- corr_CV(comb.obj, 'Cluster1', 'Cluster3', 10, rep=1000)
mac.cd8 <- tibble('ctrl' = ctrl.mac.cd8$corr, 
                  'icb' = icb.mac.cd8$corr,
                  'comb' = comb.mac.cd8$corr)
openxlsx::write.xlsx(mac.cd8, '/Users/katebridges/Downloads/20240731_mac-CD8_corr-10x10-1000.xlsx')

# cd8-treg corr
ctrl.cd8.treg <- corr_CV(ctrl.obj, 'Cluster3', 'Cluster4', 20, rep=100)
icb.cd8.treg <- corr_CV(icb.obj, 'Cluster3', 'Cluster4', 20, rep=100)
comb.cd8.treg <- corr_CV(comb.obj, 'Cluster3', 'Cluster4', 20, rep=100)
cd8.treg <- tibble('ctrl' = ctrl.cd8.treg$corr, 
                   'icb' = icb.cd8.treg$corr,
                   'comb' = comb.cd8.treg$corr)
openxlsx::write.xlsx(cd8.treg, '/Users/katebridges/Downloads/20240718_CD8-Treg_corr-20x20-100.xlsx')

# plotting mac-treg corr vs. mac-cd8 corr for same subsampled grids from comb treatment only
mac.cd8.treg <- corr_CV(comb.obj, 'Cluster1', 'Cluster3', 20, rep=1000, feature3='Cluster4')
cor.test(mac.cd8.treg$corr, mac.cd8.treg$corr.feature3)
openxlsx::write.xlsx(mac.cd8.treg, '/Users/katebridges/Downloads/20240718_3-feature_corr-20x20-1000.xlsx')

mac.cd8.treg %>% 
  ggplot(aes(x=corr, y=corr.feature3)) +
  geom_point()+ theme_bw() + 
  theme(aspect.ratio=1) +
  labs(x ='Mac-CD8+ correlation', y = 'Mac-Treg correlation') +
  geom_smooth(method = "lm")



