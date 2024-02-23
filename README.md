![logo](docs/github-header-image1.png)
## Installation
Muffin can be installed through conda. Right now it is only available on a
private channel and does not work on windows, but it is planned to migrate to
conda-forge or bioconda. The extra channels are required for dependencies. For
this reason we highly recommend starting from a fresh environment (and
optionally to use mamba instead of conda, which resolves conflicts faster and
better than conda): 
```sh
# Change ENV_NAME to the name of your choice, and add OTHER_PACKAGES if needed
conda install -c conda-forge mamba
mamba create -n ENV_NAME -c pdelangen13 -c bioconda -c conda-forge muffin OTHER_PACKAGES
conda activate ENV_NAME
```
## Documentation
Guidelines, examples and API reference are available on
[ReadTheDocs](http://muffin.readthedocs.io/).

## Reproduce the examples from the paper
Clone the repository : 
```sh
git clone https://github.com/benoitballester/Pol2Atlas.git
```
Download the data from [Zenodo](https://doi.org/10.5281/zenodo.10561582) (~5GB uncompressed): 
```sh
wget https://zenodo.org/records/10561582/files/muffin_examples_data.tar.gz?download=1
tar -zxvf muffin_examples_data.tar.gz
```

Set the path to the data in settings/paths.py (rootFolder variable), and use the
notebooks located in docs/examples/. Make sure to be in the repository root
folder when launching notebooks, or use os.chdir() within the notebook. You can
optionally use the provided apptainer/singularity image instead of a conda
environment.

If you really wish to reproduce the H3K4Me3 ChIP-seq analysis (WARNING : it will
download around 100GB of data from ENCODE), you will need to run the the
snakemake script "dl_data.smk" located in the immune_chip/ folder downloaded
from the zenodo repository.

 
## Cite
If you use Muffin in your work, please cite :
```
MUFFIN : A suite of tools for the analysis of functional sequencing data
Pierre de Langen, Benoit Ballester
bioRxiv 2023.12.11.570597; doi: https://doi.org/10.1101/2023.12.11.570597
```