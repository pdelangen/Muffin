.. Muffin documentation master file, created by
   sphinx-quickstart on Thu Jul  6 11:08:30 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. image:: github-header-image1.png
  :width: 90%
  :align: center

| 

Welcome to MUFFIN's documentation!
==================================
MUFFIN is a python package that offers multiple tools for the analysis of count-based functional genomic data.
Its flexible count-modelling approach allows to analyze a wide variety of assays of different sizes and sequencing depths, 
from the bulk to single-cell level.
It also includes tools to link genomic regions to genes and functional annotations. Finally,
MUFFIN integrates seamlessly into the `Scanpy <https://github.com/scverse/scanpy>`_ ecosystem. 

.. toctree::
   :maxdepth: 2
   :caption: How to use MUFFIN

   Using MUFFIN

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/H3K4Me3_chip_immune.ipynb
   examples/ATAC_TCGA.ipynb
   examples/10k_pbmc_clustering.ipynb

.. toctree::
   :maxdepth: 1
   :caption: API reference

   muffin

   