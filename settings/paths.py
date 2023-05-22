rootFolder = "/shared/projects/pol2_chipseq/data_newPkg/"
# rootFolder = "/home/delangen/ifbProject/data_newPkg/"
# genome annotation
ctcf_remap_nr = rootFolder + "genome_annot/remap2022_CTCF_nr_macs2_hg38_v1_0.bed"
chromsizes = rootFolder + "genome_annot/hg38.chrom.sizes.sorted"
GOfile = rootFolder + "GO_files/hsapiens.GO:BP.name.gmt"
gencode = rootFolder + "genome_annot/gencode.v38.annotation.gtf"
# scATAC
scAtacHD5 = rootFolder + "scATAC/atac_v1_pbmc_10k_filtered_peak_bc_matrix.h5"
scAtacMapqual= rootFolder + "scATAC/atac_v1_pbmc_10k_singlecell.csv"
# scRNA-seq
scRNAseqPol2 = rootFolder + "10k_pbmc_pol2_atlas/filtered_feature_bc_matrix/"
scRNAseqGenes = rootFolder + "10k_pbmc_gene/filtered_feature_bc_matrix/"
# immune chip
immuneChipPath = rootFolder + "immune_chip/"
# 
chipBAM = rootFolder + "bam_chip/chip/"
inputBAM = rootFolder + "bam_chip/input/"
chipmetadata = rootFolder + "bam_chip/metadata.tsv"
chipmetadata_exp = rootFolder + "bam_chip/metadata_exp.tsv"
chipBAM2 = rootFolder + "bam_chip/ENCFF555KAJ.bam"

