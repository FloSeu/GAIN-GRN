GAIN-GRN: A Generic Residue Numbering Scheme for GPCR Autoproteolysis Inducing (GAIN) Domains
===================================================================================

|Python| |Jupyter| |DOI| |Zenodo| |License|

``GAIN-GRN`` is a Python module that provides the full workflow for establishing a generic residue numbering (GRN) scheme with a pre-calculated dataset of PDB structures, based on detection and filtering
of GAIN domains, determining the variance of the GAIN domain dataset and establishing templates for structural alignments alongside the secondary structural elements and their centers.

It is the package behind the `GPCRdb <https://www.gpcrdb.org>`_ 's adhesion GPCR (aGPCR) nomenclature functionality, introduced in this study:

* Generic residue numbering of the GAIN domain of adhesion GPCRs
    | Florian Seufert, Guillermo Pérez-Hernández, Gáspár Pándy-Szekeres, Ramon Guixà-González, Tobias Langenhan, David E. Gloriam, Peter W. Hildebrand
    | ReasearchSquare
    | https://doi.org/10.21203/rs.3.rs-4761600/v1

**TLDR** to locally replicate what the above functionality does online, and tag a pdbfile with GRN labels, we have provided a Jupyter notebook to dynamically assign the GAIN-GRN labels to any GAIN domain, including domains of the related Polycystic Kidney disease /
Polycystic Kidney disease-like (PKD1/PKD1L1) GAIN domains.

In order to run the workflow locally, you have to install the third party dependencies `GESAMT <https://www.ccp4.ac.uk/download/>`_ via ccp4, `STRIDE <https://webclu.bio.wzw.tum.de/stride/install.html>`_ as well as `curl <https://curl.se/>`_ for retrieving `UniProt <https://www.uniprot.org>`_ and `AlphaFold data <https://alphafold.ebi.ac.uk/>`_ data. For other uses, please refer to the `Usage Guide <GUIDE.rst>`_ and `FAQ <FAQ.rst>`_.


Installation
============

Clone the repository, cd into the ``GAIN-GRN`` folder and install via ``pip`` 

.. code:: bash

   pip install gaingrn

To be able to use the package, notebooks and tests, please use the ``gaingrn.utils.io.download_data()`` function to download the necessary data from the `zenodo repository <https://dx.doi.org/10.5281/zenodo.12515545>`_.
in your cloned repository, use *python* with:

.. code:: python

   import gaingrn.utils.io
   gaingrn.utils.io.download_data()

Furthermore, if you want to execute *GESAMT* and *STRIDE*, you will need the corresponding binaries and set them via:

.. code:: bash

   export GESAMT_BIN=/path/to/your/GESAMT
   export STRIDE_BIN=/path/to/your/STRIDE

Here are links for installing `GESAMT <https://www.ccp4.ac.uk/download/>`_ via ccp4 and `STRIDE <https://webclu.bio.wzw.tum.de/stride/install.html>`_.

The dynamic GAIN-GRN assignment notebook uses ``curl`` as a dependency for retrieving UniProt and AlphaFold data. Ensure it is runnable from command line.

Licenses
========
* ``GAIN-GRN`` is licensed under the `GNU Lesser General Public License v3.0 or later <https://www.gnu.org/licenses/lgpl-3.0-standalone.html>`_ (``LGPL-3.0-or-later``, see the LICENSE.txt).

* Modules used by ``GAIN-GRN`` have different licenses. You can check any module's license in your Python environment using `pip-licenses <https://github.com/raimon49/pip-licenses>`_:

.. code:: bash

   pip-licenses | grep module_name

Documentation
=============
Please refer to the `Usage Guide <GUIDE.rst>`_ and `FAQ <FAQ.rst>`_

System Requirements
===================
``GAIN-GRN`` is developed in GNU/Linux. Tested Python versions are:

* GNU/Linux: 3.9, 3.10

Authors
=======
``GAIN-GRN`` is written and maintained by  Florian Seufert (`ORCID <http://orcid.org/0000-0002-0664-7169>`_) currently at the `Institute of Medical Physics and Biophysics <https://biophysik.medizin.uni-leipzig.de/>`_ in the
`Universität Leipzig <https://www.uni-leipzig.de/>`_.

Please cite:
 * Generic residue numbering of the GAIN domain of adhesion GPCRs
    | Florian Seufert, Guillermo Pérez-Hernández, Gáspár Pándy-Szekeres, Ramon Guixà-González, Tobias Langenhan, David E. Gloriam, Peter W. Hildebrand
    | ReasearchSquare
    | https://doi.org/10.21203/rs.3.rs-4761600/v1

Status
======
``GAIN-GRN`` is approaching its release alongside publication.

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

 .. |Zenodo| image:: 
    https://zenodo.org/badge/DOI/10.5281/zenodo.12515545.svg
    :target: https://doi.org/10.5281/zenodo.12515545