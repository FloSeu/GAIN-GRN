GAIN-GRN: A Generic Residue Numbering Scheme for GPCR Autoproteolysis Inducing (GAIN) Domains
===================================================================================

|Python| |Jupyter| |DOI| |License|

..
   .. figure:: doc/imgs/banner.png
      :scale: 33%

   .. figure:: doc/imgs/distro_and_violin.png
      :scale: 25%

   .. figure:: doc/imgs/timedep_ctc_matrix.png
      :scale: 55%

   .. figure:: doc/imgs/interface.combined.png
      :scale: 33%

``GAIN-GRN`` is a Python module that provides the full workflow for establishing a generic residue numbering (GRN) scheme with a pre-calculated dataset of PDB structures, based on detection and filtering
of GAIN domains, determining the variance of the GAIN domain dataset and establishing templates for structural alignments alongside the secondary structural elements and their centers. With a complete set of
template structures, their segments and their segment centers, a dynamic notebook is provided to dynamically assign the GAIN-GRN to any GAIN domain, including domains of the related Polycystic Kidney disease / 
Polycystic Kidney disease-like (PKD1/PKD1L1) GAIN domains.

Licenses
========
* ``GAIN-GRN`` is licensed under the `GNU Lesser General Public License v3.0 or later <https://www.gnu.org/licenses/lgpl-3.0-standalone.html>`_ (``LGPL-3.0-or-later``, see the LICENSE.txt).

* Modules used by ``GAIN-GRN`` have different licenses. You can check any module's license in your Python environment using `pip-licenses <https://github.com/raimon49/pip-licenses>`_:

  >>> pip-licenses | grep module_name

Documentation
=============
Please refer to this readme and the code annotations for dcoumentations.

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

 .. |Python Package| image::
    https://github.com/FloSeu/GAIN-GRN/figures/python39.svg
 
 .. |Jupyter| image::
    https://github.com/FloSeu/GAIN-GRN/figures/jupyterlab.svg

 .. |License| image::
    https://github.com/FloSeu/GAIN-GRN/figures/gpl3.svg
    :target: https://github.com/FloSeu/GAIN-GRN/LICENSE.txt
 
 .. |DOI| image::
    https://img.shields.io/badge/DOI-10.21203%2Frs.3.rs--4761600%2Fv1-blue
    :target: https://doi.org/10.21203/rs.3.rs-4761600/v1