GAIN-GRN: A Generic Residue Numbering Scheme for GPCR Autoproteolysis Inducing (GAIN) Domains
===================================================================================

**Can this approach be generalized to any protein domain**

In general, **yes**. Look at the workflow presented in `GUIDE.rst <GUIDE.rst>`_: With the GAIN domains, we are dealing with a heterogenous and (so far) poorly annotated protein domain. We can easily
use the *Uniprot Domain* entries of better conserved domains, i.e. EGF, to extract a set of structures from the AlphaFoldDB which is more consistent as a base dataset. In this case, the following procedure is advised:

1. Write a separate detection function specific for the key markers of your domain
   - Identify key regions of high/low conservation and structural homo-/heterogeneity.
   - Be aware of possible subdomains that need splitting for templates.

2. Create a new *DomainObject* class which has key markers, boundaries etc. of your target domain
   - In case of GAIN domains, we are dealing with domains that have *either* helices or strands. In mixed domains, the data structure needs to be altered to accurately reflect the secondary structural composition

3. Adjust subselection routines for template detection and curation
   
   - Are there different families of proteins in your dataset? You should assess the structural variance in each of subselections until you reach a sufficient level of coverage.
   - Assign names to segments manually in your *template data*. For this, you can either do a "maximalist approach", where you assign names to as many segments as possible, regardless of conservation; 
      or you stick to a "minimalist approach", where only the most conserved segments are labeled.

**Are cancer-enriched positions all relevant for wet-lab experiments?**

The analysis of cancer enrichment provides only a correlation to cancer, not a causation. The datasets provided in the `Cancer Genome Atlas <https://portal.gdc.cancer.gov>`_ only map variances of cancer 
patients. Therefore, we cannot reliably assess that this mutation *causes* cancer, only that it might be associated. The associated impact scores like `SIFT <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC168916>`_
and `Polyphen <getetics.bwh.harvard.edu/pph2/>`_ try to assess the mutational impact by approximating the physicochemical impact of an amino-acid substutition. Therefore, we advise to use the position-specific
analysis tools provided in `6 Cancer Mutation Analysis <https://github.com/FloSeu/GAIN-GRN/blob/main/gaingrn/6_cancer_analysis.ipynb>`_ to assess all natural and cancer variants for your GRN-labeled position of interest, 
before using expensive wet-lab workflows to test.