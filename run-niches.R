# libraries

library(Seurat)
library(ggplot2)
library(dplyr)
library(scales)

# library(devtools)
# install_github('msraredon/NICHES', ref = 'master')
# install_github('saezlab/OmnipathR')

library(NICHES)
library(stringr)
library(ggrepel)

# use of h5seurat formatting to convert sc obj between python and R

run_niches <- function(adata) {
  
  # split adata into indiv objects per condition + replicate
  data.list <- SplitObject(adata, split.by="sample_rep")
  
  # for each division of data (condition + replicate),
  # impute with alra & generate NICHES network 
  imp.list <- list()
  for(i in 1:length(data.list)){
    imputed <- SeuratWrappers::RunALRA(data.list[[i]])
    imp.list[[i]] <- RunNICHES(imputed,
                               LR.database="omnipath",
                               species="mouse",
                               assay="alra",
                               cell_types = "grouping",
                               meta.data.to.map = c('sample','celltype','grouping'),
                               SystemToCell = T,
                               CellToCell = T)
  }
  names(imp.list) <- names(data.list)
  
  # retain only cell2cell signaling output
  temp.list <- list()
  for(i in 1:length(imp.list)){
    temp.list[[i]] <- imp.list[[i]]$CellToCell 
    temp.list[[i]]$Condition <- names(imp.list)[i] # Tag with metadata
  }
  
  # merge list back into single object
  merged.list <- temp.list[[1]]
  for (k in length(temp.list)-1) {
    merged.list <- merge(merged.list, temp.list[[k+1]])
  }
  
  # limiting to connections with > 5 links (optional)
  adata.sub <- subset(merged.list, nFeature_CellToCell > 5)
  
  # Perform initial visualization
  adata.sub <- ScaleData(adata.sub)
  adata.sub <- FindVariableFeatures(adata.sub,selection.method = "disp")
  adata.sub <- RunPCA(adata.sub,npcs = 50)
  adata.sub <- RunUMAP(adata.sub,dims = 1:25)
  # DimPlot(adata.sub,group.by = c('Condition', 'VectorType'))
  
  return(adata.sub)
}


