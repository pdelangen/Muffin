"""
Genomic Regions Enrichment Analysis
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pyranges as pr
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from statsmodels.stats.multitest import fdrcorrection
from scipy.sparse import csr_array
import muffin


from .utils import cluster, overlap_utils, utils, stats

maxCores = len(os.sched_getaffinity(0))

class regLogicGREAT:
    def __init__(self, upstream, downstream, distal):
        self.upstream = upstream
        self.downstream = downstream
        self.distal = distal

    def __call__(self, txDF, chrInfo):
        # Infered regulatory domain logic
        copyTx = txDF.copy()
        copyTx["Start"] = (txDF["Start"] - self.upstream).where(txDF["Strand"] == "+", 
                                    txDF["End"] - self.downstream)
        copyTx["End"] = (txDF["Start"] + self.downstream).where(txDF["Strand"] == "+", 
                                    txDF["End"] + self.upstream)
        copyTx.sort_values(["Chromosome", "Start"], inplace=True)
        try:
            copyTx["Chromosome"].cat.remove_unused_categories(inplace=True)
        except:
            pass
        gb = copyTx.groupby("Chromosome")
        copyTx["Chromosome"] = copyTx["Chromosome"]
        perChr = dict([(x,gb.get_group(x)) for x in gb.groups])
        for c in perChr:
            inIdx = copyTx["Chromosome"] == c
            nextReg = np.roll(copyTx["Start"][inIdx], -1)
            try:
                nextReg[-1] = chrInfo.loc[c].values[0]
            except:
                print(f"Warning: chromosome '{c}' in gtf but not in size file, skipping all genes within this chromosome.")
                copyTx = copyTx[np.logical_not(inIdx)]
                continue
            previousReg = np.roll(copyTx["End"][inIdx], 1)
            previousReg[0] = 0
            extMin = np.maximum(copyTx["Start"][inIdx] - self.distal, previousReg)
            extMax = np.minimum(copyTx["End"][inIdx] + self.distal, nextReg)
            extMin = np.minimum(copyTx["Start"][inIdx], extMin)
            extMax = np.maximum(copyTx["End"][inIdx], extMax)
            copyTx.loc[copyTx["Chromosome"] == c, "Start"] = np.clip(extMin, 0, chrInfo.loc[c].values[0])
            copyTx.loc[copyTx["Chromosome"] == c, "End"] = np.clip(extMax, 0, chrInfo.loc[c].values[0])
        return copyTx


class pyGREAT:
    def __init__(self, gmtFile, geneFile, chrFile, 
                 distal=1000000, upstream=5000, downstream=1000, 
                 gtfGeneCol = "gene_name", gene_biotype="all"):
        """
        Genomic regions GSEA tool

        Parameters
        ----------
        gmtFile : str or list of str
            Path to gmt file or list of paths to gmt files that will get 
            concatenated.
        geneFile : str
            Path to a GTF file.
        chrFile : str
            Path to a chrInfo file. (chrom-tab-Chrom size-line return)
        distal : int, optional
            Size of inferred distal regulatory regions, by default 1000000
        upstream : int, optional
            Size of inferred upstream regulatory regions, by default 5000
        downstream : int, optional
            Size of inferred downstream regulatory regions, by default 1000
        gtfGeneCol : str, optional
            Name of the gene name/id column in the GTF file, by default "gene_name"
        gene_biotype : str, optional
            Type of gene to keep e.g. "protein_coding", by default "all"
        """
        self.chrInfo = pd.read_csv(chrFile, sep="\t", index_col=0, header=None)
        self.gtfGeneCol = gtfGeneCol
        # Parse GMT file
        # Setup gene-GO matrix
        if type(gmtFile) is str:
            gmtFile = [gmtFile]
        self.all_associations = list()
        self.goMap = dict()
        self.allGenes = set()
        for gmtF in gmtFile:
            with open(gmtF) as f:
                for l in f:
                    vals = l.rstrip("\n").split("\t")
                    for g in vals[2:]:
                        self.all_associations.append((vals[0], g))
                    self.allGenes |= set(vals[2:])
                    self.goMap[vals[0]] = vals[1]
        self.allGenes = list(self.allGenes)
        self.allTerms = list(self.goMap.keys())
        self.invGoMat = {v: k for k, v in self.goMap.items()}
        # Read gtf file
        self.transcripts = pr.read_gtf(geneFile, as_df=True)
        self.transcripts = self.transcripts[self.transcripts["Feature"] == "gene"]
        if not gene_biotype == "all":
            self.transcripts = self.transcripts[self.transcripts["gene_type"] == gene_biotype]
        self.transcripts = self.transcripts[["Chromosome", "Start", "End", self.gtfGeneCol, "Strand"]]
        # Apply infered regulatory logic
        self.distal = distal
        self.upstream = upstream
        self.downstream = downstream
        self.geneRegulatory = regLogicGREAT(upstream, downstream, distal)(self.transcripts, self.chrInfo)
        self.geneRegulatory.drop("Strand", 1, inplace=True)
        self.geneRegulatory.index = self.geneRegulatory["gene_name"]
        self.geneRegulatory = self.geneRegulatory[~self.geneRegulatory.index.duplicated(False)]
        self.validGenes = self.geneRegulatory["gene_name"]
        self.geneRegulatory = self.geneRegulatory.loc[self.validGenes]
        # Transform gene set - gene associations to a sparse binary matrix (for clustering and glm)
        goFa, gos = pd.factorize(np.array(self.all_associations)[:,0])
        geneFa, genes = pd.factorize(np.array(self.all_associations)[:,1])
        non_annotated_genes = np.setdiff1d(self.validGenes.values, genes)
        genes = np.concatenate([genes, non_annotated_genes])
        data = np.ones_like(goFa, dtype="bool")
        self.mat = pd.DataFrame.sparse.from_spmatrix(csr_array((data, (geneFa, goFa)), shape=(len(genes), len(gos))).T, 
                                                     columns=genes, index=gos)



    def getNearestGene(self, query, max_dist=np.inf):
        """
        Get the nearest gene for each row. 
        If it can't be retrieved will be named "None" (str).

        Parameters
        ----------
        query : pandas Dataframe or PyRanges
            _description_
        
        max_dist : integer or float
            Maximum distance for association, default np.inf

        Returns
        -------
        _type_
            _description_
        """
        if not isinstance(query, pd.DataFrame):
            query = query.as_df()
        gr = query.copy()
        gr["Names"] = np.arange(len(gr))
        gr = pr.PyRanges(gr[["Chromosome", "Start", "End", "Names"]])
        gr2 = pr.PyRanges(self.transcripts[["Chromosome", "Start", "End", "gene_name"]])
        r = gr.nearest(gr2, strandedness=False)
        chroms_missing = set(gr.chromosomes).difference(gr2.chromosomes)
        if len(chroms_missing) > 0:
            missing_gr = pr.concat([gr[c] for c in chroms_missing]).as_df()
            missing_gr["gene_name"] = "None"
            missing_gr = pr.PyRanges(missing_gr)
            results = pr.concat([r,  missing_gr], strand=False).as_df()
        else:
            results = r.as_df()
        results.index=results["Names"]
        results = results.loc[np.arange(len(gr))]
        results["gene_name"].where(results["Distance"] < max_dist, "None", inplace=True)
        return results
    
    def label_by_nearest_gene(self, query):
        """
        Label each row by its nearest gene, avoiding duplicates.
        E.g. NADK_1, MYC_1, MYC_2, None_1, None_2

        Parameters
        ----------
        query : pandas Dataframe or PyRanges
            Set of genomic regions.

        Returns
        -------
        labels: ndarray
            Each corresponding label.
        """
        names = self.getNearestGene(query, max_dist=self.distal)
        groups = names.groupby('gene_name')
        # add a count for each duplicate value within each group
        names['count'] = groups.cumcount() + 1
        # create new column with unique names for each duplicate value
        names['col1_new'] = names['gene_name'] + '_' + names['count'].astype(str)
        # drop the count column and the original column with duplicate values
        names = names.drop(['gene_name', 'count'], axis=1)
        return names["col1_new"].values

    def findEnrichedGenes(self, query, background=None):
        """
        Find enriched terms in genes near query.

        Parameters
        ----------
        query: pandas dataframe in bed-like format or PyRanges
            Set of genomic regions to compute enrichment on.
        background: None, pandas dataframe in bed-like format, or PyRanges (default: None)
            If set to None considers the whole genome as the possible locations of the query.
            Otherwise it supposes the query is a subset of these background regions.
        
        Returns
        -------
        results: pandas dataframe
        """
        from rpy2.robjects.packages import importr
        rs = importr("stats")
        regPR = pr.PyRanges(self.geneRegulatory.rename({self.gtfGeneCol:"Name"}, axis=1))
        refCounts = overlap_utils.countOverlapPerCategory(regPR, 
                                                          overlap_utils.dfToPrWorkaround(background, useSummit=False))
        allCats = np.array(list(refCounts.keys()))
        pvals = np.zeros(len(allCats))
        fc = np.zeros(len(allCats))
        M = len(background)
        # Then for the query
        obsCounts = overlap_utils.countOverlapPerCategory(regPR, 
                                                          overlap_utils.dfToPrWorkaround(query, useSummit=False))
        N = len(query)
        # Find hypergeometric enrichment
        k = pd.Series(np.zeros(len(allCats), dtype="int"), allCats)
        isFound = np.isin(allCats, obsCounts.index, assume_unique=True)
        k[allCats[isFound]] = obsCounts
        n = pd.Series(np.zeros(len(allCats), dtype="int"), allCats)
        n[allCats] = refCounts
        # Scipy hyper 
        pvals = np.array(rs.phyper(k.values-1,n.values,M-n.values,N, lower_tail=False))
        pvals = np.nan_to_num(pvals, nan=1.0)
        fc = (k/np.maximum(N, 1e-7))/np.maximum(n/np.maximum(M, 1e-7), 1e-7)
        qvals = fdrcorrection(pvals)[1]
        pvals = pd.Series(pvals)
        pvals.index = allCats
        qvals = pd.Series(qvals)
        qvals.index = allCats
        fc = pd.Series(fc)
        fc.index = allCats
        geneEnriched = pvals, fc, qvals, k, n
        geneEnriched = pd.DataFrame(geneEnriched, 
                            index=["P-value", "FC", "FDR", "Query hits", "Background hits"]).T
        return geneEnriched.sort_values("P-value")
        
    
    def findGenesForGeneSet(self, term, enrichedGeneTab, alpha=0.05):
        """
        Find genes enriched for a particular geneset

        Parameters
        ----------
        term : str
            Name of the GO term.
        enrichedGeneTab : pandas dataframe
            Results of findEnrichedGenes.
        alpha : float, optional
            Adjusted P-value threshold, by default 0.05

        Returns
        -------
        Enriched: pandas Dataframe
            Enriched genes
        """
        sigGenes = enrichedGeneTab[enrichedGeneTab["FDR"] < alpha].index
        genesInTerm = self.mat.columns[self.mat.loc[term] > 0.5]
        enrichedInTerm = sigGenes.intersection(genesInTerm)
        return enrichedGeneTab.loc[enrichedInTerm]




    def findEnriched(self, query, background=None, expected_mappability=0.85, minGenes=3, maxGenes=1000, cores=-1):
        """
        Find enriched terms in genes near query.

        Parameters
        ----------
        query: pandas dataframe in bed-like format or PyRanges
            Set of genomic regions to compute enrichment on. If a pandas dataframe,
            assume the first three columns are chromosome, start, end.
        background: None, pandas dataframe in bed-like format, or PyRanges (default: None)
            If set to None considers the whole genome as the possible, equiprobable
            locations of the query.
            Otherwise it supposes the query is a subset of these background regions. 
            (Note that it does not explicitly check for it).
        expected_mappability: float
            Effective only when background is set to None.
            Multiplier for the total genome size to correct non-mappable regions.
        minGenes: int, (default 3)
            Minimum number of intersected genes for a GO annotation.
        maxGenes: int, (default 1000)
            Maximum number of genes for a GO annotation.
        cores: int, (default -1)
            Max number of cores to used for parallelized computations. Default uses all
            available cores (-1).
        
        Returns
        -------
        results: pandas dataframe
        """
        # First compute intersections count for each gene
        # And expected intersection count for each gene
        regPR = pr.PyRanges(self.geneRegulatory.rename({self.gtfGeneCol:"Name"}, axis=1))
        if background is not None:
            intersectBg = overlap_utils.countOverlapPerCategory(regPR, overlap_utils.dfToPrWorkaround(background, useSummit=False))
        else:
            genomeSize = np.sum(self.chrInfo).values[0]*expected_mappability
            intersectBg = (self.geneRegulatory["End"]-self.geneRegulatory["Start"])/genomeSize
            intersectBg = np.maximum(intersectBg, 1/genomeSize)
            intersectBg.index = self.geneRegulatory["gene_name"]
            intersectBg = intersectBg.groupby(intersectBg.index).sum()
        intersectQuery = overlap_utils.countOverlapPerCategory(regPR, overlap_utils.dfToPrWorkaround(query, useSummit=False))
        queryCounts = intersectBg.copy() * 0
        queryCounts.loc[intersectQuery.index] = intersectQuery
        obsMatrix = self.mat[queryCounts.index]
        if background is not None:
            expected = intersectBg.loc[obsMatrix.columns]
            observed = pd.DataFrame(queryCounts.loc[queryCounts.index])
            endog = observed.copy()
            expected *= len(query)/len(background)
        else: 
            expected = intersectBg.loc[obsMatrix.columns]*len(query)
            observed = pd.DataFrame(queryCounts.loc[queryCounts.index])
            endog = observed.copy()
        # Trim GOs under cutoff
        trimmed = obsMatrix[intersectQuery.index].sum(axis=1) >= minGenes
        trimmed = trimmed & (obsMatrix.sum(axis=1) <= maxGenes)
        # Setup parallel computation settings
        if cores == -1:
            cores = maxCores      
        maxBatch = len(obsMatrix.loc[trimmed])
        maxBatch = min(int(0.25*maxBatch/cores)+1,256)
        hitsPerGO = np.sum(csr_array(obsMatrix.sparse.to_coo()) * observed.values.ravel(), axis=1)[trimmed.values]
        # Fit a Negative Binomial GLM for each annotation, and evaluate wald test p-value for each gene annotation
        with Parallel(n_jobs=cores, batch_size=maxBatch, max_nbytes=None, mmap_mode=None) as pool:
            results = pool(delayed(stats.fitNBinomModel)(hasAnnot, endog, expected, gos, queryCounts.index) for gos, hasAnnot in obsMatrix[trimmed].iterrows())
        # Manually kill workers afterwards or they'll just stack up with multiple runs
        get_reusable_executor().shutdown(wait=False, kill_workers=True)
        # Format results
        results = pd.DataFrame(results)
        if results.shape[1] > 0:
            results.set_index(0, inplace=True)
            results.columns = ["P(Beta > 0)", "Beta"]
        else :
            results = pd.DataFrame(None, columns=["P(Beta > 0)", "Beta"])
        results.dropna(inplace=True)
        results["P(Beta > 0)"] = np.maximum(results["P(Beta > 0)"], 1e-320)
        qvals = results["P(Beta > 0)"].copy()
        qvals.loc[:] = fdrcorrection(qvals.values)[1]
        results["BH corrected p-value"] = qvals
        results["-log10(qval)"] = -np.log10(qvals)
        results["-log10(pval)"] = -np.log10(results["P(Beta > 0)"])
        results["FC"] = np.exp(results["Beta"])
        results["Name"] = [self.goMap[i] for i in results.index]
        results["Total hits"] = hitsPerGO
        results.sort_values(by="P(Beta > 0)", inplace=True)
        return results

    def plotEnrichs(self, enrichDF, title="", by="P(Beta > 0)", alpha=0.05, topK=10, savePath=None):
        """
        Draw Enrichment barplots

        Parameters
        ----------
        enrichDF: pandas dataframe or tuple of pandas dataframes
            The result of the findEnriched method.
            
        savePath: string (optional)
            If set to None, does not save the figure.
        """
        fig, ax = plt.subplots(figsize=(2,2),dpi=muffin.params["figure_dpi"])
        newDF = enrichDF.copy()
        newDF.index = [self.goMap[i] for i in newDF.index]
        selected = (newDF["BH corrected p-value"] < alpha)
        ordered = -np.log10(newDF[by][selected]).sort_values(ascending=True)[:topK]
        terms = ordered.index
        t = [utils.capTxtLen(term, 50) for term in terms]
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(length=3, width=1.2)
        ax.barh(range(len(terms)), np.minimum(ordered[::-1],324.0))
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(t[::-1], fontsize=5)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel("-log10(Corrected P-value)", fontsize=8)
        ax.set_title(title, fontsize=10)
        if muffin.params["autosave_plots"] is not None:
            fig.savefig(muffin.params["autosave_plots"] + "/barplot_enrich" + muffin.params["autosave_format"], bbox_inches="tight")
        return fig, ax

    def clusterTreemap(self, enrichDF, alpha=0.05, score="-log10(qval)", 
                       metric="yule", resolution=1.0, output=None):
        """
        Plot a treemap of clustered gene set terms.
        Clustering is done according to similarities between annotated genes
        of a gene set.

        Parameters
        ----------
        enrichDF : dataframe
            Output of "findEnriched" method.
        alpha : float, optional
            FDR cutoff, by default 0.05
        score : str, optional
            Which column of the result dataframe to use to identify lead
            term, by default "-log10(qval)"
        metric : str, optional
            Similarity measure between GO terms, by default "yule"
        resolution : float, optional
            Influences the size of the clusters (larger = more clusters), by default 1.0
        output : str or None, optional
            Path to save figure, by default None
        """
        sig = enrichDF[enrichDF["BH corrected p-value"] < alpha]
        if len(enrichDF) == 0:
            return
        # Remove unused columns and keep enriched rows
        nz = csr_array(self.mat.loc[sig.index].sparse.to_coo()).sum(axis=0) > 1
        simplifiedMat = self.mat.loc[sig.index, nz].sparse.to_dense().values
        clusters = cluster.graphClustering(simplifiedMat, 
                                           metric, k=int(np.sqrt(len(sig))), r=resolution, snn=True, 
                                           approx=True, restarts=10)
        sig.loc[:,"Cluster"] = clusters
        sig.loc[:,"Name"] = [utils.customwrap(self.goMap[i]) for i in sig.index]
        representatives = pd.Series(dict([(i, sig["Name"][sig[score][sig["Cluster"] == i].idxmax()]) for i in np.unique(sig["Cluster"])]))
        sig.loc[:,"Representative"] = representatives[sig["Cluster"]].values
        duplicate = sig["Representative"] == sig["Name"]
        sig.loc[:, "Representative"][duplicate] = ""
        fig = px.treemap(names=sig["Name"], parents=sig["Representative"], 
                        values=sig[score],
                        width=800, height=800)
        fig.update_layout(margin = dict(t=2, l=2, r=2, b=2),
                        font_size=30)
        fig.show()
        if output is not None:
            fig.write_image(output)
            fig.write_html(output + ".html")
        if muffin.params["autosave_plots"] is not None:
            fig.write_image(muffin.params["autosave_plots"] + "/barplot_enrich" + muffin.params["autosave_format"])
            fig.write_html(muffin.params["autosave_plots"] + "/barplot_enrich" + muffin.params["autosave_format"] + ".html")




    
