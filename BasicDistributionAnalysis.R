
# Uploading library and files ---------------------------------------------

library(igraph)
library(reticulate)
library(biclust)
library(dplyr)
library(pheatmap)
library(gridExtra)
library(RColorBrewer)
library(VennDiagram)
library(ggVennDiagram)
library(ggplot2)
#Importing all pickle files for top 10
pd <- import("pandas")
grp1_grp2_sub <- pd$read_pickle("DataMatrixGrp2_Grp1.pickle")
grp1_sub <- pd$read_pickle("DataMatrixGrp1.pickle")
originalsnp <- pd$read_pickle("DataMatrixOg.pickle")
top_snp_ids = names(grp1_grp2_sub)


# Distribution plots ------------------------------------------------------
# Obtaining the Brain Features from all features info
all_features = originalsnp['rs429358'][1][[1]]$xlabels
features_imp = c('Gliobla','Astrocytes','Monocytes','Brain','Neuro')
brain_related_features = c()
brain_related_features_boolean = matrix(0,1,2002)
for (f in features_imp){
  idx_feats = grepl(f,all_features)
  brain_related_features_boolean = brain_related_features_boolean + as.numeric(idx_feats)
  brain_related_features = c(brain_related_features, unlist(all_features[idx_feats]))
  print(paste0("Total number of features for ",f," is ",as.character(sum(idx_feats))))
}
# removing featal indexes
featal_idx = grepl('Fetal',all_features)
brain_related_features_boolean[featal_idx] = 0
brain_related_features = brain_related_features[!grepl('Fetal',brain_related_features)]


# Main SNP distribution
plot_distribution_mainsnp = function(snp_name,brain_related = FALSE){
  title = paste0("Distribution of MainSNP only: ",snp_name)
  if (brain_related){
    t = grp1_sub[[snp_name]]$data
    t = t[as.logical(brain_related_features_boolean)]
    title = paste0('Brain Related Regions Distribution - ',snp_name)
    absolute_values = abs(as.numeric(as.matrix(t)))
  }else{  
    title = paste0('Whole Distribution - ',snp_name)
    absolute_values = abs(as.numeric(as.matrix(grp1_sub[[snp_name]]$data)))
  }
  
  return(hist(absolute_values,breaks = 100,  main =snp_name))
}





# Combination Max SNP distribution
plot_distribution_combsnps = function(snp_name,brain_related = FALSE){
  
  title = paste0("Distribution of MainSNP only: ",snp_name)
  group2_data_matrix = grp2_whole[snp_name][1][[1]]$data
  if (brain_related){
    t = group2_data_matrix
    t = t[,as.logical(brain_related_features_boolean)]
    title = paste0('Brain Related Regions Distribution - ',snp_name)
    absolute_max_values = apply(abs(t),2,max)
  }else{  
    title = paste0('Whole Distribution - ',snp_name)
    absolute_max_values = apply(abs(group2_data_matrix),2,max)
  }
  return(hist(absolute_max_values,breaks = 100,  main =snp_name))
}

# Combination Max SNP - Absolute Ind distribution
plot_distribution_diff = function(snp_name,brain_related= FALSE){
  absolute_values = abs(as.matrix(originalsnp[snp_name][1][[1]]$data)[1,as.logical(brain_related_features_boolean)])
  title = paste0("Distribution of Difference: ",snp_name)
  group2_data_matrix = grp2_whole[snp_name][1][[1]]$data[,as.logical(brain_related_features_boolean)]
  if (brain_related){
    absolute_values = abs(as.matrix(originalsnp[snp_name][1][[1]]$data)[1,as.logical(brain_related_features_boolean)])
    title = paste0("Distribution of Difference: ",snp_name)
    group2_data_matrix = grp2_whole[snp_name][1][[1]]$data[,as.logical(brain_related_features_boolean)]
  }else{  
    absolute_values = abs(as.matrix(originalsnp[snp_name][1][[1]]$data)[1,])
    title = paste0("Distribution of Difference: ",snp_name)
    group2_data_matrix = grp2_whole[snp_name][1][[1]]$data
  }
  absolute_max_values = apply(abs(group2_data_matrix),2,max)
  res = absolute_values - absolute_max_values
  return(hist(res,breaks = 100,  main =snp_name))
}



## Distribution Plots of Main SNP Only

par(mfrow = c(2,5))
for (name in top_snp_ids){
  plot_distribution_mainsnp(name,brain_related = TRUE)
}


## Distribution Plots of Comb SNP Only

par(mfrow = c(2,5))
for (name in top_snp_ids){
  plot_distribution_combsnps(name,brain_related = TRUE)
}


## Distribution Plots of Difference Only

par(mfrow = c(2,5))
for (name in top_snp_ids){
  plot_distribution_diff(name,brain_related = TRUE)
}


