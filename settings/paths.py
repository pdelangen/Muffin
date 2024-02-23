rootFolder = "/shared/projects/pol2_chipseq/data_newPkg/"
# rootFolder = "X:/data_newPkg/"
# rootFolder = "/home/delangen/ifbProject/data_newPkg/"
# genome annotation
chromsizes = rootFolder + "genome_annot/hg38.chrom.sizes.sorted"
GOfile = rootFolder + "GO_files/hsapiens.GO:BP.name.gmt"
gencode = rootFolder + "genome_annot/gencode.v38.annotation.gtf"
gencodehg19 = rootFolder + "genome_annot/gencode.v45lift37.basic.annotation.gtf"
# scATAC
scAtacHD5 = rootFolder + "scATAC/atac_v1_pbmc_10k_filtered_peak_bc_matrix.h5"
scAtacMapqual= rootFolder + "scATAC/atac_v1_pbmc_10k_singlecell.csv"
# Cancer hallmarks
atac_cancer_meta = rootFolder + "tcga_atac/sequencing_stats.csv"
atac_cancer_table = rootFolder + "tcga_atac/atac_table.txt"
cancerHallmarks = rootFolder + "tcga_atac/hallmark_cancer_chg.gmt"
# scRNA-seq
scRNAseqGenes = rootFolder + "10k_pbmc_gene/filtered_feature_bc_matrix/"
sct_clusters = rootFolder + "10k_pbmc_gene/cluster_labels.csv"
# immune chip
immuneChipPath = rootFolder + "immune_chip/"

# Peakmerge benchmarks
ctcf_remap_bed = rootFolder + "genome_annot/remap2022_CTCF_all_macs2_hg38_v1_0.bed"
cage_peaks = rootFolder + "genome_annot/cage_peaks/"
histone_peaks = rootFolder + "genome_annot/histone_peaks/"