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
library(Seurat)
library(SeuratData)
library(Matrix)
library(stringr)

get_gene_UMI_count <- function(file_dir, csv_name) {
  to.return <- c()
  ##read in the coordinates of points lying on top of the tissue.position.txt is generated from matlab script "Pixel_identification.m". 
  location1 <- read.table(file.path(file_dir,"position.txt"), sep =",", header = FALSE, dec =".", stringsAsFactors = F)
  
  ##read expression matrix and generate the Filtered_matrix.tsv, which contains only the useful pixels
  # first col is xy location, colnames are gene names
  my_data <- read.table(file = file.path(file_dir, csv_name), sep = ',', header = FALSE, 
                        stringsAsFactors=FALSE)
  my_data <- t(my_data) 
  my_data <- my_data %>% row_to_names(row_number = 1)
  data_filtered <- my_data[my_data[,1] %in% location,] # for ICB only sample, removes ~500 pixels
  rownames(data_filtered) <- data_filtered[,1]
  data_filtered <- data_filtered[, -1]
  mode(data_filtered) <- 'numeric'
  
  ##calculate the total UMI count and Gene count
  UMI_count <- rowSums(data_filtered)
  data_filtered_binary <- as_tibble(data_filtered) %>% dplyr::mutate_all(as.logical)
  gene_count <- rowSums(data_filtered_binary)
  
  to.return$data <- data_filtered
  to.return$UMI.count <- UMI_count
  to.return$gene.count <- gene_count
  
  return(to.return)
}

generate_Seurat_obj <- function(file_dir, sample.name, tsv.name, cluster.res, sample, sept = "", transp = TRUE) {
  # read in per pixel counts 
  if (transp) {
    expr_sample <-t(as.matrix(read.table(file = file.path(file_dir, tsv.name))))
  } else {
    expr_sample <-as.matrix(read.table(file = file.path(file_dir, tsv.name), sep=',', header=TRUE))
    expr_sample <- expr_sample %>% as.data.frame() %>% column_to_rownames(var = "X")
  }
  
  if (startsWith(colnames(expr_sample)[1], 'X')) {
    colnames(expr_sample) <- str_sub(colnames(expr_sample), 2)
  }
  
  # Remove (non-coding gene?) rows 
  expr_sample <- expr_sample[!grepl("^Gm", rownames(expr_sample)), ]
  expr_sample <- expr_sample[!grepl("^ENSMUS", rownames(expr_sample)), ]
  expr_sample <- expr_sample[!grepl("^RNA", rownames(expr_sample)), ]
  
  pos_sample <- fread(file.path(file_dir,"position.txt"),header = F) %>% t
  # pos_sample = pos_sample[!is.na(pos_sample),]
  pos_sample <- rbind(pos_sample, c("1x1"))
  pos_sample <- rbind(pos_sample, c("50x50"))
  pos_sample <- rbind(pos_sample, c("1x50"))
  pos_sample <- rbind(pos_sample, c("50x25"))
  pos_sample <- rbind(pos_sample, c("25x50"))
  pos_sample <- rbind(pos_sample, c("50x1"))
  pos_sample <- rbind(pos_sample, c("25x1"))
  pos_sample <- rbind(pos_sample, c("1x25"))
  pos_sample_df <-
    data.frame (pos = pos_sample) %>% filter(!is.na(pos)) %>% distinct() %>%
    mutate(
      xcoord = gsub('x.*$', '', pos) %>% as.numeric(),
      ycoord = gsub('.*x', '', pos)  %>% as.numeric()
    ) %>%
    set_rownames(.$pos) %>%
    dplyr::select(xcoord, ycoord)
  tmp <- pos_sample_df$xcoord
  pos_sample_df$xcoord <-  pos_sample_df$ycoord
  pos_sample_df$ycoord <-  tmp
  
  # pos and expr crossover
  pos_cor <- expr_sample %>% colnames() %>%
    .[. %in%  rownames(pos_sample_df)] %>% as.character()
  pos_sample_df_cor <- pos_sample_df[pos_cor, ]
  expr_sample_cor <- expr_sample[, pos_cor]
  
  # create seurat object
  samp <-
    CreateSeuratObject(counts = expr_sample_cor,
                       project = sample,
                       assay = 'Spatial')
  samp$slice <- 1
  samp$region <- sample
  samp[[sample]] <- new(Class = 'SlideSeq', assay = "Spatial",coordinates = pos_sample_df_cor)
  samp <- subset(samp, subset = nFeature_Spatial > 1)
  
  samp@meta.data %<>% mutate(UMIs = nCount_Spatial)
  samp@meta.data %<>% mutate(Genes = nFeature_Spatial)
  
  #normalize with STtransform 
  samp <- SCTransform(samp, assay = "Spatial", verbose = FALSE, min_cells = 2,
                      return.only.var.genes = FALSE) #include all genes set to FALSE 
  samp@meta.data %<>% mutate(UMIs_SCT = nCount_SCT)
  samp@meta.data %<>% mutate(Genes_SCT = nFeature_SCT)
  
  # DIMENSIONAL REDUCTION AND CLUSTERING 
  samp <- RunPCA(samp, assay = "SCT", verbose = FALSE)
  samp <- FindNeighbors(samp, reduction = "pca", dims = 1:30)
  samp <- FindClusters(samp, verbose = FALSE, resolution = cluster.res)
  samp <- RunUMAP(samp, reduction = "pca", dims = 1:30)
  
  
  return(samp)
}

corr_CV <- function(obj, feature1, feature2, sample_size, min.pix=1.33, rep=100, feature3='none') {
  # want to sample sample_size x sample_size grid from 50x50 grid 
  limit = 50 - sample_size
  min.pixels <- (sample_size*sample_size)/min.pix
  
  xy <- c()
  corr.master <- c()
  pval.master <- c()
  
  corr.feature3 <- c()
  pval.feature3 <- c()
  
  for (g in 1:rep) {
    # randomly generate lower LH coordinate and find sample grid corner indices
    useful.pixels <- 0
    # using while loop to confirm that sampled grid contains at least 1/3 useful pixels
    while(useful.pixels < min.pixels) {
      x = sample(1:limit, 1)
      y = sample(1:limit, 1)
      x.end = x + sample_size
      y.end = y + sample_size
      
      # generate list of coordinates within sample grid
      coord <- c()
      for (xind in c(x:x.end)) {
        for (yind in c(y:y.end)) {
          coord <- c(coord, paste0(yind, 'x', xind))
        }
      }
      
      # match to what exists in 50x50 grid
      tissue.coord <- rownames(GetTissueCoordinates(obj)[,c('x','y')])
      coord_filtered <- intersect(coord, tissue.coord)
      useful.pixels <- length(coord_filtered)
      # print(useful.pixels)
    }
    
    xy <- c(xy, paste0(y, 'x', x))
    
    # evaluate & add to master
    test <- cor.test(obj[[feature1]][coord_filtered,], obj[[feature2]][coord_filtered,])
    
    corr.master <- c(corr.master, test$estimate)
    pval.master <- c(pval.master, test$p.value)
    
    if (feature3 != 'none') {
      test2 <- cor.test(obj[[feature1]][coord_filtered,], obj[[feature3]][coord_filtered,])
      corr.feature3 <- c(corr.feature3, test2$estimate)
      pval.feature3 <- c(pval.feature3, test2$p.value)
    }
  }
  to.return <- tibble('xy' = xy, 'corr' = corr.master, 'pval' = pval.master)
  
  if (feature3 != 'none') {
    to.return <- to.return %>% add_column('corr.feature3' = corr.feature3)
    to.return <- to.return %>% add_column('pval.feature3' = pval.feature3)
  }
  
  return(to.return)
}
