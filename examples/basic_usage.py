"""
Basic Usage Example for Dynamic Causal Strength (DCS)

This example demonstrates the basic usage of the DCS package for
causal inference in time series data.
"""

import numpy as np
from dcs import generate_signals, snapshot_detect_analysis_pipeline
from dcs.config import (
    BicParams,
    CausalParams,
    DetectionParams,
    MonteCParams,
    OutputParams,
    PipelineConfig,
    PipelineOptions,
)


def main() -> None:
    """
    Demonstrate basic usage of the DCS package.
    
    This function shows how to:
    1. Generate synthetic time series data
    2. Configure the analysis pipeline
    3. Run the causal analysis
    4. Access and interpret results
    """
    print("Dynamic Causal Strength (DCS) - Basic Usage Example")
    print("=" * 60)

    # Step 1: Generate synthetic data
    print("\n1. Generating synthetic time series data...")
    data = generate_synthetic_data()
    print(f"Generated data shape: {data.shape}")

    # Step 2: Prepare signals for analysis
    print("\n2. Preparing signals for analysis...")
    original_signal, detection_signal = prepare_signals(data)
    print(f"Original signal shape: {original_signal.shape}")
    print(f"Detection signal shape: {detection_signal.shape}")

    # Step 3: Configure analysis pipeline
    print("\n3. Configuring analysis pipeline...")
    config = create_pipeline_config()
    print("Configuration created successfully.")

    # Step 4: Run analysis
    print("\n4. Running causal analysis...")
    results = run_analysis(original_signal, detection_signal, config)

    # Step 5: Display results
    print("\n5. Analysis results:")
    display_results(results)


def generate_synthetic_data() -> np.ndarray:
    """
    Generate synthetic time series data for demonstration.
    
    Returns
    -------
    np.ndarray
        Synthetic data with shape (2, T, Ntrial)
    """
    # Parameters for synthetic data generation
    T = 1200  # Total time points
    Ntrial = 20  # Number of trials
    h = 0.1  # Time step
    gamma1 = 0.5  # Damping coefficient 1
    gamma2 = 0.5  # Damping coefficient 2
    Omega1 = 1.0  # Natural frequency 1
    Omega2 = 1.2  # Natural frequency 2

    # Generate coupled oscillator signals
    data, _, _ = generate_signals(
        T=T,
        Ntrial=Ntrial,
        h=h,
        gamma1=gamma1,
        gamma2=gamma2,
        Omega1=Omega1,
        Omega2=Omega2,
    )
    
    return data


def prepare_signals(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare signals for causal analysis.
    
    Parameters
    ----------
    data : np.ndarray
        Raw time series data with shape (2, T, Ntrial)
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        original_signal: Trial-averaged signal
        detection_signal: Signal for event detection
    """
    # Compute trial-averaged signal
    original_signal = np.mean(data, axis=2)
    
    # Create detection signal (amplified version for event detection)
    detection_signal = original_signal * 1.5
    
    return original_signal, detection_signal


def create_pipeline_config() -> PipelineConfig:
    """
    Create configuration for the analysis pipeline.
    
    Returns
    -------
    PipelineConfig
        Configured pipeline settings
    """
    return PipelineConfig(
        options=PipelineOptions(
            detection=True,
            bic=False,
            causal_analysis=True,
            bootstrap=False,
            save_flag=False,
            debiased_stats=False,
        ),
        detection=DetectionParams(
            thres_ratio=2.0,
            align_type="peak",
            l_extract=150,
            l_start=75,
            remove_artif=True,
        ),
        bic=BicParams(morder=4),
        causal=CausalParams(ref_time=75, estim_mode="OLS"),
        monte_carlo=MonteCParams(n_btsp=50),
        output=OutputParams(file_keyword="example_run"),
    )


def run_analysis(
    original_signal: np.ndarray,
    detection_signal: np.ndarray,
    config: PipelineConfig,
) -> dict:
    """
    Run the causal analysis pipeline.
    
    Parameters
    ----------
    original_signal : np.ndarray
        Original time series data
    detection_signal : np.ndarray
        Signal used for event detection
    config : PipelineConfig
        Pipeline configuration
    
    Returns
    -------
    dict
        Analysis results
    """
    try:
        # Run the complete analysis pipeline
        results, final_config, event_snapshots = snapshot_detect_analysis_pipeline(
            original_signal=original_signal,
            detection_signal=detection_signal,
            config=config,
        )
        
        print("✓ Pipeline completed successfully!")
        return results
        
    except Exception as e:
        print(f"✗ Pipeline failed with error: {e}")
        raise


def display_results(results: dict) -> None:
    """
    Display and interpret analysis results.
    
    Parameters
    ----------
    results : dict
        Results from the analysis pipeline
    """
    print(f"Number of detected events: {len(results.get('locs', []))}")
    print(f"Model order used: {results.get('morder', 'N/A')}")
    
    if results.get("CausalOutput"):
        causal_output = results["CausalOutput"]["OLS"]
        
        # Display DCS results
        if "DCS" in causal_output:
            dcs_values = causal_output["DCS"]
            print(f"DCS result shape: {dcs_values.shape}")
            print(f"Mean DCS (X→Y): {np.mean(dcs_values[:, 0]):.4f}")
            print(f"Mean DCS (Y→X): {np.mean(dcs_values[:, 1]):.4f}")
        
        # Display Transfer Entropy results
        if "TE" in causal_output:
            te_values = causal_output["TE"]
            print(f"Transfer Entropy shape: {te_values.shape}")
            print(f"Mean TE (X→Y): {np.mean(te_values[:, 0]):.4f}")
            print(f"Mean TE (Y→X): {np.mean(te_values[:, 1]):.4f}")
        
        # Display rDCS results
        if "rDCS" in causal_output:
            rdcs_values = causal_output["rDCS"]
            print(f"Relative DCS shape: {rdcs_values.shape}")
            print(f"Mean rDCS (X→Y): {np.mean(rdcs_values[:, 0]):.4f}")
            print(f"Mean rDCS (Y→X): {np.mean(rdcs_values[:, 1]):.4f}")
    else:
        print("No causal output available.")


if __name__ == "__main__":
    main()
