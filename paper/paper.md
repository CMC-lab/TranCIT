---
# JOSS Paper Draft - Revised Version
# the final release version/date, and DOI placeholders are filled in.

title: "TranCIT: Transient Causal Interaction Toolbox"
tags:
  - Python
  - neuroscience
  - causal inference
  - time series analysis
  - Granger causality
  - transfer entropy
  - Local field potential (LFP)
  - Electroencephalogram (EEG)
  - Magnetoencephalography (MEG)
authors:
  - name: Salar Nouri
    affiliation: 1
    orcid: 0000-0002-8846-9318
    corresponding: true
  - name: Kaidi Shao
    affiliation: "2"
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
csl: ieee.csl
repository: "https://github.com/cmc-lab/trancit"
crossref: true
doi: "10.5281/zenodo.16998396" # Zenodo DOI for the archived version
url: "https://trancit.readthedocs.io/en/latest/" 

---

## Summary

The study of complex systems, e.g., neural circuits and cognitive functions, often requires understanding causal interactions that occur during brief, transient events [@logothetisHippocampalCorticalInteraction2012; @womelsdorfBurstFiringSynchronizes2014; @nitzanBrainwideInteractionsHippocampal2022; @safaviBrainComplexSystem2022; @safaviUncoveringOrganizationNeural2023; @lundqvistBetaBurstsCognition2024]. Traditional methods for estimating causality, such as Granger causality [@granger1969investigating] and Transfer Entropy (TE) [@Schreiber2000], are frequently challenged by the short-duration, non-stationary nature of these phenomena [@mitra2007observed]. Such methods typically assume stationarity and require long data segments, making them suboptimal for event-driven analysis.

We present `trancit` (Transient Causal Interaction Toolbox), a robust, open-source Python package implementing advanced causal inference methods specifically designed to address this challenge [@nouri_2025_trancit_package; @nouri2025trancit]. `trancit` provides a comprehensive pipeline for dynamic causal inference on multivariate time-series data. It is a Python-native implementation and extension of a powerful causal learning algorithm originally introduced in MATLAB [@shao2023transient]. By leveraging foundational Python libraries like NumPy [@harris2020array] and SciPy [@virtanen2020fundamental], `trancit` integrates seamlessly into modern data science and research workflows, making these advanced techniques accessible to a broader scientific community.

The package offers an integrated solution for end-to-end causal analysis, including:

- **Advanced causal inference methods:** A suite of algorithms to detect and quantify causal relationships, including classic methods like Granger Causality (GC) and Transfer Entropy (TE), alongside the more robust, Structural Causal Model (SCM) based Dynamic Causal Strength (DCS) and relative Dynamic Causal Strength (rDCS).

- **Event-based preprocessing:** An automated pipeline to identify transient event timings from signals, align data epochs relative to these events, and reject trials contaminated by artifacts.

- **Simulation and validation tools:** Utilities to generate synthetic autoregressive (AR) time-series data with known ground-truth causal structures, enabling users to validate methods, test hypotheses, and explore theoretical scenarios in a controlled environment.

## Statement of need

While many statistical methods focus on correlation, the ability to infer directed causal relationships offers deeper, more mechanistic insights into how complex systems function [@Seth2015]. A critical frontier in this field is the analysis of transient dynamics, where interactions can rapidly change or occur in brief, intense bursts. While powerful methods to analyze these dynamics have been developed, their implementation in proprietary software like MATLAB [@shao2023transient] has limited their accessibility and adoption within the broader open-source scientific ecosystem.

`trancit` bridges this critical gap by providing a fully open-source, Python-based implementation. While general-purpose Python causality libraries such as `causal-learn` [@zheng2024causal] and `tigramite` [@runge2022jakobrunge] are invaluable, they often lack the specialized features required for robust analysis of transient, event-related data. For example, they may not offer integrated workflows for event detection and alignment, which are crucial for this type of analysis.

`trancit` offers a tailored solution that implements GC, TE, DCS, and rDCS with configurations specifically suited for potentially non-stationary signals. By providing these tools in an accessible Python package, `trancit` promotes reproducible research, lowers the barrier to entry for advanced causal inference, and supports a wide range of applications in fields such as neuroscience, climatology, and economics.

## Functionality

### Causal inference methods

`trancit` implements four primary methods for detecting and quantifying causal relationships. A brief overview is provided here; for full mathematical derivations and theoretical background, please refer to our main methodology paper [@shao2023transient; @nouri2025trancit].

- **Granger Causality (GC):** A classic and widely used method based on vector autoregressive models that assesses whether the history of one time series improves the prediction of another.

- **Transfer Entropy (TE):** A non-parametric, information-theoretic measure that quantifies the directed flow of information and reduction of uncertainty between two signals.

- **Dynamic Causal Strength (DCS):** A robust method grounded in the framework of SCMs. DCS quantifies time-varying causal influence through a principled interventional approach. This allows DCS to overcome common failure modes of other methods, such as the "synchrony pitfall," where measures like TE can be misleading during periods of high signal synchronization.

- **relative Dynamic Causal Strength (rDCS):** An extension of DCS specifically tailored for event-based analysis. It quantifies causal effects relative to a pre-defined baseline or reference period, making it exceptionally sensitive to the deterministic shifts in signal dynamics that often characterize event-related data.

#### Preprocessing and simulation

A key feature of `trancit` is its integrated workflow for preparing time-series data for causal analysis. The preprocessing module includes functions for detecting event timings based on signal thresholds, aligning data epochs to these events (e.g., to a local signal peak), and automatically extracting data segments or "snapshots" for analysis. The package also includes a simulation module to generate synthetic AR data with user-defined causal structures. This feature is invaluable for validating the methods, for educational purposes, and for exploring the sensitivity of the causal measures under different conditions (e.g., varying noise levels or data lengths).

## Example

To demonstrate `trancit`'s core functionality and validate its implementation, we replicated key results from @shao2023transient ([Figure 4](https://www.frontiersin.org/files/Articles/1085347/fnetp-03-1085347-HTML-r1/image_m/fnetp-03-1085347-g004.jpg)). As shown in \autoref{fig:causality}, our simulation example highlights the "synchrony pitfall" of Transfer Entropy. DCS correctly identifies a continuous underlying causal link in a system with a transient period of high synchrony, a scenario where TE incorrectly suggests that the causal link has weakened or vanished.

![Replication of @shao2023transient Figure 4 using `trancit` package. Shows successful detection of directed influence from X to Y using simulated data and causality measures (e.g., TE, DCS) implemented in the package. \label{fig:causality}](figures/3_dcs_example.pdf "Figure 1: Causality detection on simulated data")

To demonstrate its utility on real-world scientific data, we used `trancit` to analyze publicly available Local Field Potential (LFP) recordings from the rodent hippocampus during sharp wave-ripple events. As shown in **\autoref{fig:ca1_ca3_analysis}**, the rDCS method correctly identifies the well-established transient information flow from hippocampal area CA3 to CA1. Critically, this result underscores the importance of proper experimental design, as the causal link is only apparent when the analysis is correctly aligned on the putative cause (CA3), a key feature facilitated by our package.

![Demonstration of `trancit` on real-world LFP data showing directed causality from hippocampal area CA3 to CA1. The analysis successfully identifies transient information flow during sharp-wave ripple events using the package's built-in rDCS method. \label{fig:ca1_ca3_analysis}](figures/4_ca3_ca1_analysis.pdf "Figure 2: Event-based causal analysis of hippocampal LFP data. The plot shows a transient increase in directed influence from CA3 to CA1, computed using rDCS.")

## Implementation details

The `trancit` package is open-source and distributed under the permissive **BSD-2 license**, ensuring it can be freely used, modified, and incorporated into diverse academic and commercial projects [@nouri_2025_trancit_package; @nouri2025trancit]. The package is designed with usability and extensibility in mind, featuring a highly modular architecture that separates concerns for causality, modeling, simulation, and utilities. It includes robust error handling for invalid configurations and relies on a comprehensive test suite implemented with the `pytest` framework to ensure algorithmic reliability. Continuous integration is managed via GitHub Actions. The project welcomes community involvement, and detailed guidelines for contributing code or reporting issues are provided in the software repository.

## Acknowledgments

We acknowledge the foundational work by Kaidi Shao, Nikos Logothetis, and Michel Besserve [@shao2023transient] on the dynamic causal strength methodology. We also thank the developers and communities behind the core Python scientific libraries utilized in `trancit`, including NumPy [@harris2020array], SciPy [@virtanen2020fundamental], and Matplotlib [@hunter2007matplotlib]. KS acknowledges the support from the Shanghai Municipal Science and Technology Major Project (Grant No. 2019SHZDZX02) and the Max Planck Society (including the Max Planck Institute for Biological Cybernetics and the Graduate School of Neural and Behavioral Sciences).
SS acknowledges the support from Max Planck Society and add-on fellowship from the Joachim Herz Foundation.

## References
