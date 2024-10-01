GAIN-GRN: A Generic Residue Numbering Scheme for GPCR Autoproteolysis Inducing (GAIN) Domains
===================================================================================

This documentation contains explanations for the contents and usage of the Jupyter notebooks and scripts contained in the package.

.. |nb_url| replace:: github.com

Preface
-------

For the notebooks to work, we need the GAINGRN data downloaded via:

.. code:: python

   gaingrn.scripts.io.download_data()

and for the notebook 1 and 2 to function, the set of aGPCR GAIN PDB models is needed. It can be downloaded via:

.. code:: python

   gaingrn.scripts.io.download_pdbs(target_directory='path/to/your/PDB_DIR')

Both archives can also be retrieved manually from the `zenodo repository <https://dx.doi.org/10.5281/zenodo.12515545>`_

|JLogo| `(0) Filter Stage <https://github.com/gph82/GAIN-GRN/blob/main/gaingrn/notebooks/0_filter_stage.ipynb>`_
-----------------------------------------------------------------------------------------------------

*This notebook is mainly for documentation purposes, since it is for pre-filtering the 3D models before the analysis.*
Missing data for re-running this notebook can be requested from the author if desired.

Here, the GAIN domain 3D models and their secondary structure data evaluated by `STRIDE <https://webclu.bio.wzw.tum.de/stride/>`_ are filtered according to GAIN domain detection criteria:
   
   - Presence of a 20+ resiude Subdomain A
   - Presence of a "GPS" or analogous stretch of residues between the last two beta-strands

The valid GAIN domains are collected into a *GainCollection* object containing their N-, C- and subdomain boundary as well as information about their secondary structural elements.
It is stored in a PKL file.

A similar procedure is done for PKD1/PKD1L1 GAIN domain models in |JLogo| `pkd_gain/pkd_gain_processing.ipynb <gaingrn/pkd_gain/pkd_gain_processing.iypnb>`_.

|JLogo| `1 Template Selection <https://github.com/gph82/GAIN-GRN/blob/main/gaingrn/notebooks/1_template_selection.ipynb>`_
=================================================================================================================

Two sets of templates exist for each respective subdomain to account for their different degrees of conservation. The template selection and following curation workflow look as follows:

.. figure:: figures/workflow.png
   :width: 400

Subselections are created based on manually defined criteria (Here, we use aGPCR homologs, i. e. "ADGRA2") and each subselection is aligned to itself via GESAMT, creating an n² RMSD matrix.
The subselection is then assessed `agglomerative clustering <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>`_ to see the variance within the subselection.
If the variance is sufficiently small and coverage is decent, the centroid of the largest cluster is selected as a *potential template*. Every *potential template* is pairwise-matched against the whole 
*valid collection* and MSAs are created for each segment by stacking the pairwise residue matches. From these MSAs, the segment center can be extracted by finding the highest occupancy and conservation
in the segment resdiues.

|JLogo| `2 Template Curation <https://github.com/gph82/GAIN-GRN/blob/main/gaingrn/notebooks/2_template_curation.ipynb>`_
===============================================================================================================

The previously defined *potential templates* with their matches are assessed for coverage. For this, evaluate each recpeptor subselectionand use the respective sets of pairwise alignments to the 
*potential template* to assess:
   
   - the fraction of the models that have matches with the segment and segment center
   - the distance of the pairwise segment center match

.. figure:: figures/l-match.png
   :width: 600

Not all GAIN models have for example six helices in Subdomain A, evidence by the **X** markers ni the figure. Assessing all templates, each receptor homolog has a template assigned for each subdomain that
covers all segments with sufficient quality (see in `template_data.json <data/template_data.json>`_).

*If running the template curation, you need the folders for a complete matching of the dataset against each template via pairwise GESAMT. Run these via

|JLogo| `3 Assign the GAIN-GRN <https://github.com/gph82/GAIN-GRN/blob/main/gaingrn/notebooks/3_assign_gaingrn.ipynb>`_
==============================================================================================================

After having the template PDB files and the respective information on segments and segment centers, we can now create a full GRN assignment of all GAIN domain models. In this notebook, GAIN-GRN is
assigned to specific *GainDomain* objects or to a whole *GainCollection*. For faster assignment, a parallelized version is available: |PLogo| `run_indexing.py <run_indexing.py>`_. The full set of alignment data
is stored in a pickle file (`data/stal_indexing.pkl <data/stal_indeixng.pkl>`_)

|JLogo| `4 GAIN-GRN Dataset Statistics and Analysis <https://github.com/gph82/GAIN-GRN/blob/main/gaingrn/notebooks/4_gaingrn_statistics.ipynb>`_
=======================================================================================================================================

The complete set of GRN assignments can now be used to statistically evaluate the GAIN domain model dataset. Here, conservation per residue and segment occupation are assessed (see Figure 2 and Supp. Fig. 1).

|JLogo| `5 Cancer Mutation Analysis <https://github.com/gph82/GAIN-GRN/blob/main/gaingrn/notebooks/5_cancer_analysis.ipynb>`_
====================================================================================================================

As an application for the GAIN-GRN, the natural variant and cancer mutation data for the human aGPCR can be mapped to their respective GRN labels. This enables the calculation of a cancer enrichment score
analogous to `Wright et al. (2019) <ttps://www.nature.com/articles/s41467-019-08630-2>`_, enabling the identification of hotspots of cancer-enriched residues (see Figure 5, Supp. Fig. 2). Here, also tools are provided to extract the full variant/mutation information
for every labeled position (which receptor, which substitution, which potential impact) for the individual assessment of residues for designing wet-lab experiments. 

|JLogo| `Dynamic GAIN-GRN Assignment <https://github.com/gph82/GAIN-GRN/blob/main/gaingrn/notebooks/dynamic_gain_grn.ipynb>`_
====================================================================================================================

Here, we provide a complete workflow to dynamically assign the GAIN-GRN to any GAIN-domain containing protein. By just providing the UniProtKB identifier, the `UniProt <https://www.uniprot.org>`_ 
information and the 3D model from `AlphaFoldDB <https://alphafold.ebi.ac.uk>`_ are retrieved and automatically assessed. The notebook guides the user through the GAIN-GRN assignment process. This is
especially useful for distantly related proteins, i.e. PKD1/PKD1L1 proteins or invertebrate aGPCRs.

Please also refer to the `FAQ <FAQ.rst>`_ for further info.

 .. |PLogo| image:: 
   https://github.com/FloSeu/GAIN-GRN/blob/main/figures/plogo.png
   :height: 2ex
   :class: no-scaled-link

 .. |JLogo| image:: 
   https://github.com/FloSeu/GAIN-GRN/blob/main/figures/jlogo.png
   :height: 2ex
   :class: no-scaled-link

 .. |Python| image::
    https://github.com/FloSeu/GAIN-GRN/blob/main/figures/python39.svg

 .. |Jupyter| image::
    https://github.com/FloSeu/GAIN-GRN/blob/main/figures/jupyterlab.svg

 .. |License| image::
    https://github.com/FloSeu/GAIN-GRN/blob/main/figures/gpl3.svg
    :target: https://github.com/FloSeu/GAIN-GRN/LICENSE.txt
 
 .. |DOI| image::
    https://img.shields.io/badge/DOI-10.21203%2Frs.3.rs--4761600%2Fv1-blue
    :target: https://doi.org/10.21203/rs.3.rs-4761600/v1

