# Generating Paper Figures

This document explains how to reproduce the figures shown in the TranCIT software paper using the provided example scripts.

## Figure 1: Causality Detection on Simulated Data

**Figure Location:** `paper/figures/3_dcs_example.pdf`

**Description:** This figure replicates Figure 4 from Shao et al. (2023), demonstrating the "synchrony pitfall" where Transfer Entropy (TE) fails during high-synchronization periods, while Dynamic Causal Strength (DCS) correctly identifies the underlying causal link.

### Generation Steps

1. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook examples/dcs_introduction.ipynb
   ```

2. **Run all cells** in the notebook. The notebook will:
   - Generate synthetic coupled oscillator time series data
   - Run multiple simulation runs (typically 20 runs) to ensure statistical robustness
   - Compute time-varying causality measures (TE, DCS, GC) for each run
   - Average results across runs
   - Generate plots showing:
     - Example time series signals
     - Noise profiles
     - Causality measures over time (TE, DCS, GC)

3. **Save the figure:** The notebook generates the plot inline. To save it as a PDF for the paper:
   ```python
   plt.savefig('paper/figures/3_dcs_example.pdf', dpi=300, bbox_inches='tight')
   ```

## Figure 2: Event-Based Causal Analysis of Hippocampal LFP Data

**Figure Location:** `paper/figures/4_ca3_ca1_analysis.pdf`

**Description:** This figure demonstrates TranCIT's application to neural data, showing transient information flow from hippocampal area CA3 to CA1 during sharp wave-ripple events.

### Generation Steps

The figure is generated in three sequential steps:

#### Step 1: Compute Causality Analysis

Run the analysis script to process the LFP data and compute causality measures:

```bash
python examples/compute_ca3_ca1_causality.py
```

**What this script does:**
- Loads CA3 and CA1 LFP signals from .mat files (CRCNS HC-3 dataset)
- Applies bandpass filtering (140-230 Hz) to extract ripple frequency band
- Detects sharp wave-ripple events using the pipeline's event detection
- Runs causality analysis (TE, DCS, rDCS) for each channel pair
- Saves results as .npz files for visualization

**Output files:**
- `ca3_ca1_{sess_name}_chpair_{n}_{align}_model_causality.npz`
- Additional bootstrap files if bootstrap analysis is enabled

#### Step 2: Plot Causality Results

Generate the main causality analysis figure:

```bash
python examples/plot_ca3_ca1_dcs_results.py
```

**What this script does:**
- Loads the saved causality results from Step 1
- Generates publication-quality plots showing:
  - Event waveforms aligned by CA3 and CA1 peaks
  - Causality measures (TE, DCS, rDCS) for different alignment conditions
  - Statistical summaries across trials

**Output files:**
- `event_waveforms.{svg,pdf}`
- `causality_results.{svg,pdf}`

#### Step 3: Plot Example Signals

To visualize the raw signals and detected events:

```bash
jupyter notebook examples/plot_ca3_ca1_example_signals.ipynb
```

**What this notebook does:**
- Displays example LFP signals from CA3 and CA1
- Shows detected sharp wave-ripple events
- Visualizes the event detection process