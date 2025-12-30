---
title: "TranCIT: Transient Causal Interaction Toolbox"
tags:
  - Python
  - neuroscience
  - causal inference
  - time series analysis
  - Local field potential (LFP)
  - Electroencephalogram (EEG)
  - Magnetoencephalography (MEG)
authors:
  - name: Salar Nouri
    affiliation: 1
    orcid: 0000-0002-8846-9318
    corresponding: true
  - name: Kaidi Shao
    affiliation: 2
    orcid: 0000-0002-3027-0090
    corresponding: true
  - name: Shervin Safavi
    affiliation: "3, 4"
    orcid: 0000-0002-2868-530X
    corresponding: true
affiliations:
  - name: School of Electrical and Computer Engineering, College of Engineering, University of Tehran, Tehran, Iran
    index: 1
    ror: "05vf56z40"
  - name: International Center for Primate Brain Research (ICPBR), Center for Excellence in Brain Science and Intelligence Technology (CEBSIT), Chinese Academy of Sciences (CAS), Shanghai, China
    index: 2
    ror: "00vpwhm04"
  - name: Computational Neuroscience, Department of Child and Adolescent Psychiatry, Faculty of Medicine, Technische Universität Dresden, Dresden 01307, Germany
    index: 3
    ror: "042aqky30"
  - name: Department of Computational Neuroscience, Max Planck Institute for Biological Cybernetics, Tübingen 72076, Germany
    index: 4
    ror: "026nmvv73"

date: "2025-08-30"
bibliography: paper.bib
repository: "https://github.com/cmc-lab/trancit"
crossref: true
doi: "10.5281/zenodo.16998396"
url: "https://trancit.readthedocs.io/en/latest/" 

---

## Summary

The study of complex systems, particularly neural circuits and cognitive functions, requires understanding causal interactions during brief, transient events [@logothetisHippocampalCorticalInteraction2012; @womelsdorfBurstFiringSynchronizes2014; @nitzanBrainwideInteractionsHippocampal2022; @safaviBrainComplexSystem2022; @safaviUncoveringOrganizationNeural2023; @lundqvistBetaBurstsCognition2024]. Traditional causality methods, such as Granger causality (GC) [@granger1969investigating] and Transfer Entropy (TE) [@Schreiber2000], assume stationarity and require long data segments, making them suboptimal for event-driven analysis [@mitra2007observed].

We present `trancit` (Transient Causal Interaction Toolbox), an open-source Python package for causal inference in multivariate time series, emphasising the quantification of directed causal interactions during transient dynamics[@nouri_2025_trancit_package; @nouri2025trancit]. TranCIT provides a comprehensive pipeline for dynamic causal analysis, translating the robust causal learning algorithm originally introduced in MATLAB [@shao2023transient] to Python and extending it with a modular pipeline architecture, improved error handling, and enhanced integration with modern data science workflows. Built on NumPy [@harris2020array] and SciPy [@virtanen2020fundamental], TranCIT integrates seamlessly into modern data science workflows.

The package offers an integrated solution for causal effect estimation and analysis, including:

- **Advanced causal analysis methods:** GC, TE, robust Structural Causal Model (SCM)-based Dynamic Causal Strength (DCS), and relative Dynamic Causal Strength (rDCS).
- **Event-based preprocessing:** Automated event detection, data alignment, and artifact rejection pipeline.
- **Simulation tools:** Synthetic autoregressive (AR) time-series data generation with known causal structures for validation and exploring scenarios.

TranCIT primarily estimates directed causal effects in multivariate time series. Because these effects are zero when no causation exists, testing against zero also enables causal discovery. However, our primary focus is quantifying and tracking these effects over time, not proposing a discovery method.

## Statement of need

While many statistical methods focus on correlation, the ability to infer directed causal relationships offers deeper, more mechanistic insights into how complex systems function [@Seth2015]. A critical challenge is analyzing transient dynamics where interactions occur in brief, intense bursts. Existing methods are primarily implemented in proprietary software, such as MATLAB [@shao2023transient], which limits accessibility.

TranCIT bridges this gap with a fully open-source Python implementation. While packages like `statsmodels` [@seabold2010statsmodels] and `TransferEntropy` [@TransferEntropy_pkg] offer standard methods (GC and TE), and general-purpose libraries such as `causal-learn` [@zheng2024causal] and `tigramite` [@runge2022jakobrunge] focus on causal discovery, they all lack specialized features for analyzing transient, event-related data—specifically, integrated event detection and alignment workflows.

Furthermore, unlike discovery libraries that often assume stationary dynamics, TranCIT provides a tailored solution that implements GC, TE, DCS, and rDCS, with configurations for non-stationary signals. This promotes reproducible research, lowers barriers to advanced causal estimation and analysis, and supports applications in neuroscience, climatology, and economics.

## Functionality

### Causal estimation methods

TranCIT employs four primary methods to detect and quantify causal relationships. A brief overview is provided here; for complete mathematical derivations and theoretical background, please refer to our main methodology papers [@shao2023transient; @nouri2025trancit].

- **Granger Causality (GC):** Vector autoregressive model-based method assessing whether the history of one time series improves the prediction of another.
- **Transfer Entropy (TE):** Non-parametric, information-theoretic measure quantifying directed information flow between signals, and reduction of uncertainty between two signals.
- **Dynamic Causal Strength (DCS):** SCM-based method overcoming the "synchrony pitfall" where TE fails during high synchronization periods. Since it quantifies time-varying causal influence through a principled interventional approach.
- **relative Dynamic Causal Strength (rDCS):** Event-based extension quantifying causal effects relative to baseline periods. It quantifies causal effects relative to a pre-defined baseline or reference period, making it exceptionally sensitive to the deterministic shifts in signal dynamics that often characterize event-related data.

TranCIT provides integrated preprocessing for event detection using threshold-based methods with peak or pooled alignment, data alignment, and artifact rejection, as well as simulation tools for generating synthetic AR data with known causal structures for validation and education [@nouri2025trancit][Event Detection Preprocessing](../docs/event_detection_preprocessing.rst).

## Example

We validated TranCIT by replicating key results from @shao2023transient. As shown in \autoref{fig:causality}, our simulation illustrates the "synchrony pitfall," where TE fails during high-synchronization periods, while DCS correctly identifies the underlying causal link.

![Replication of @shao2023transient Figure 4 using `trancit` package. Shows successful detection of directed influence from X to Y using simulated data and causality measures (e.g., TE, DCS) implemented in the package. \label{fig:causality}](figures/3_dcs_example.pdf "Figure 1: Causality detection on simulated data")

To demonstrate its utility on real-world scientific data, TranCIT is used to analyze hippocampal LFP recordings during sharp wave-ripple events. As shown in **\autoref{fig:ca1_ca3_analysis}**, rDCS correctly identifies transient information flow from CA3 to CA1, demonstrating the importance of proper event alignment facilitated by our package.

![Demonstration of `trancit` on real-world LFP data showing directed causality from hippocampal area CA3 to CA1. The analysis successfully identifies transient information flow during sharp-wave ripple events using the package's built-in rDCS method. \label{fig:ca1_ca3_analysis}](figures/4_ca3_ca1_analysis.pdf "Figure 2: Event-based causal analysis of hippocampal LFP data. The plot shows a transient increase in directed influence from CA3 to CA1, computed using rDCS.")

Code snippets and detailed instructions for reproducing these figures and their simulation results are available in [`examples/README.md`](../examples/README.md) in the repository.

## Implementation details

The `trancit` package is distributed under the BSD-2-Clause license. TranCIT features a modular architecture separating causality, modeling, simulation, and utilities [@nouri_2025_trancit_package; @nouri2025trancit]. It includes robust error handling with custom exceptions for input validation, computation errors, configuration issues, data corruption, and numerical convergence problems, a comprehensive `pytest` test suite, and GitHub Actions continuous integration. Detailed information about the software architecture, dependencies, and design choices is available in the package documentation (see `docs/software_architecture.rst` in the repository).

## Acknowledgments

We acknowledge the foundational work by Kaidi Shao, Nikos Logothetis, and Michel Besserve [@shao2023transient] on the dynamic causal strength methodology. Kaidi Shao (KS) acknowledges the support from the Shanghai Municipal Science and Technology Major Project (Grant No. 2019SHZDZX02) and the Max Planck Society (including the Max Planck Institute for Biological Cybernetics and the Graduate School of Neural and Behavioral Sciences).
Shervin Safavi (SS) acknowledges the support from the Max Planck Society and an add-on fellowship from the Joachim Herz Foundation.

## References
