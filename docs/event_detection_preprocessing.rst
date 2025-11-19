.. _event_detection_preprocessing:

#######################################
Event Detection Preprocessing Pipeline
#######################################

Overview
========

TranCIT provides an integrated preprocessing pipeline for event detection, data alignment, and artifact rejection. This pipeline is specifically designed for analyzing transient, event-related neural dynamics where causal interactions occur during brief, intense bursts of activity (e.g., sharp wave-ripples, beta bursts, or other transient events).

The preprocessing pipeline transforms raw continuous time-series data into aligned event trials, preparing the data for subsequent causality analysis using methods such as Dynamic Causal Strength (DCS) and relative Dynamic Causal Strength (rDCS).

Preprocessing Stages
====================

The event detection preprocessing consists of five sequential stages:

1. Event Detection
------------------

**Purpose:** Identifies transient events in the detection signal using a threshold-based approach.

**Algorithm:**

- Computes a detection threshold: ``threshold = mean(signal) + thres_ratio × std(signal)``
- Identifies all time points where the detection signal exceeds this threshold
- Applies one of two alignment methods to refine event locations:
  
  - **Peak alignment:** Refines detected locations to local peaks within a specified window, ensuring events are aligned to the maximum amplitude
  - **Pooled alignment:** Uses detected locations directly, with optional location shrinking to reduce redundancy when events are detected in close temporal proximity

**Configuration Parameters:**

- ``thres_ratio`` (float): Multiplier for standard deviation in threshold calculation (higher values = fewer events detected)
- ``align_type`` (str): Either ``'peak'`` or ``'pooled'`` alignment method
- ``l_extract`` (int): Length of event windows to extract (used for peak alignment window)
- ``shrink_flag`` (bool): Whether to apply location shrinking for pooled alignment
- ``locs`` (Optional[np.ndarray]): Pre-provided event locations (if detection is disabled)

**Output:** Array of event location indices in the original signal.

2. Border Removal
----------------

**Purpose:** Filters out events that are too close to signal boundaries to ensure complete event windows can be extracted.

**Algorithm:**

- Removes event locations where ``location < l_extract`` or ``location > signal_length - l_extract``
- Ensures that each event has sufficient data before and after its center point for complete window extraction

**Configuration Parameters:**

- ``l_extract`` (int): Minimum required window length (inherited from detection stage)

**Output:** Filtered array of event locations with border events removed.

3. Snapshot Extraction
----------------------

**Purpose:** Extracts fixed-length time windows around each aligned event location, creating a 3D array of event trials.

**Algorithm:**

- For each event location, extracts a window of length ``l_extract`` starting at offset ``l_start`` from the event center
- Creates a 3D array of shape ``(n_variables × (model_order + 1), n_time_points, n_trials)``
- Includes lagged variables up to ``model_order`` for VAR model estimation
- Handles out-of-bounds windows by filling with NaN values

**Configuration Parameters:**

- ``l_extract`` (int): Length of each extracted event window
- ``l_start`` (int): Offset from event center to start extraction (can be negative)
- ``morder`` (int): Model order for VAR estimation (determines number of lagged variables)

**Output:** 3D numpy array ``(n_variables × (model_order + 1), n_time_points, n_trials)`` containing aligned event snapshots.

4. Artifact Rejection
---------------------

**Purpose:** Optionally removes trials contaminated by artifacts or signal corruption.

**Algorithm:**

- Identifies trials where any value in the first two variables falls below a specified threshold
- Removes contaminated trials from the event data array
- Updates corresponding location indices to maintain consistency

**Configuration Parameters:**

- ``remove_artif`` (bool): Whether to enable artifact removal
- ``remove_artif_threshold`` (float): Threshold below which trials are considered artifacts (default: -15000)

**Output:** Cleaned event data array and updated location indices.

5. Statistics Computation
--------------------------

**Purpose:** Computes VAR model statistics (coefficients, covariances) from the aligned event data for subsequent causality analysis.

**Algorithm:**

- Estimates VAR model coefficients using Ordinary Least Squares (OLS) or other estimation methods
- Computes residual covariances and other statistical measures
- Prepares statistics dictionary for causality calculators

**Output:** Dictionary containing VAR model statistics required for DCS/rDCS computation.

Software Architecture
======================

Pipeline Design Pattern
-----------------------

The preprocessing pipeline is implemented using a **modular stage-based architecture** that follows the **Pipeline Pattern** and **Strategy Pattern** design principles:

Core Components
~~~~~~~~~~~~~~~

1. **``PipelineOrchestrator``** (Main Coordinator)
   
   - Coordinates all preprocessing stages sequentially
   - Manages pipeline state (dictionary passed between stages)
   - Handles error propagation and logging
   - Implements the ``BaseAnalyzer`` interface for consistency with other TranCIT components

2. **``PipelineStage``** (Abstract Base Class)
   
   - Defines the interface for all preprocessing stages
   - Provides common functionality (logging, configuration access)
   - Each stage implements ``execute(**kwargs) -> Dict[str, Any]``
   - Stages are stateless and receive configuration through constructor

3. **Individual Stage Classes**
   
   - ``InputValidationStage``: Validates input data and parameters
   - ``EventDetectionStage``: Detects and aligns events
   - ``BorderRemovalStage``: Removes border events
   - ``BICSelectionStage``: Optional model order selection
   - ``SnapshotExtractionStage``: Extracts event windows
   - ``ArtifactRemovalStage``: Removes artifact-contaminated trials
   - ``StatisticsComputationStage``: Computes VAR model statistics
   - ``CausalityAnalysisStage``: Performs causality analysis (post-preprocessing)
   - Additional stages for bootstrap analysis and output preparation

Architecture Benefits
~~~~~~~~~~~~~~~~~~~~~

- **Modularity:** Each preprocessing step is a separate, testable component
- **Flexibility:** Users can customize each stage through configuration parameters
- **Extensibility:** New preprocessing stages can be added by implementing the ``PipelineStage`` interface
- **Reproducibility:** All preprocessing steps are logged and can be traced through the pipeline state
- **Maintainability:** Clear separation of concerns makes the codebase easier to understand and modify

State Management
----------------

The pipeline uses a **state dictionary** that is passed sequentially between stages:

.. code-block:: python

   pipeline_state = {
       "original_signal": original_signal,
       "detection_signal": detection_signal,
       "locs": event_locations,           # Added by EventDetectionStage
       "event_snapshots": event_data,      # Added by SnapshotExtractionStage
       "morder": model_order,              # Added by BICSelectionStage
       "stats": statistics_dict,            # Added by StatisticsComputationStage
       # ... additional state as needed
   }

Each stage:

1. Reads required data from the state dictionary
2. Performs its processing
3. Updates the state dictionary with its outputs
4. Returns the updated state

This design ensures that stages are loosely coupled and can be easily reordered or modified without affecting other stages.

API Design
==========

Configuration-Driven Architecture
----------------------------------

All preprocessing parameters are specified through dataclass-based configuration objects, enabling type safety and clear parameter documentation:

``PipelineConfig``
~~~~~~~~~~~~~~~~~~

Main configuration container that holds all pipeline parameters:

.. code-block:: python

   @dataclass
   class PipelineConfig:
       options: PipelineOptions      # Enable/disable pipeline features
       detection: DetectionParams    # Event detection parameters
       bic: BicParams               # Model selection parameters
       causal: CausalParams          # Causality analysis parameters
       # ... additional parameter groups

``DetectionParams``
~~~~~~~~~~~~~~~~~~~

Event detection-specific parameters:

.. code-block:: python

   @dataclass
   class DetectionParams:
       thres_ratio: float                    # Threshold multiplier
       align_type: str                       # 'peak' or 'pooled'
       l_extract: int                        # Window length
       l_start: int                          # Window start offset
       shrink_flag: bool = False              # Enable location shrinking
       locs: Optional[np.ndarray] = None      # Pre-provided locations
       remove_artif: bool = False             # Enable artifact removal
       remove_artif_threshold: float = -15000 # Artifact threshold

User Interface
--------------

High-Level API (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to use the preprocessing pipeline is through the ``PipelineOrchestrator``:

.. code-block:: python

   from trancit import PipelineOrchestrator, PipelineConfig, DetectionParams, PipelineOptions

   # Create configuration
   config = PipelineConfig(
       options=PipelineOptions(detection=True, bic=True),
       detection=DetectionParams(
           thres_ratio=2.0,
           align_type='peak',
           l_extract=100,
           l_start=-50
       ),
       # ... additional configuration
   )

   # Initialize orchestrator
   orchestrator = PipelineOrchestrator(config)

   # Run complete pipeline
   result = orchestrator.run(original_signal, detection_signal)

   # Access results
   event_locations = result.results['locs']
   event_snapshots = result.event_snapshots
   causality_results = result.results.get('dcs_results')

Low-Level API (Advanced Users)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Advanced users can access individual stages directly for custom workflows:

.. code-block:: python

   from trancit.pipeline.stages import EventDetectionStage, SnapshotExtractionStage

   # Create individual stages
   detection_stage = EventDetectionStage(config)
   extraction_stage = SnapshotExtractionStage(config)

   # Execute stages manually
   state = {"detection_signal": detection_signal}
   state = detection_stage.execute(**state)
   state = extraction_stage.execute(**state)

   # Access intermediate results
   event_locations = state['locs']
   event_snapshots = state['event_snapshots']

Configuration Flexibility
--------------------------

The pipeline supports multiple usage modes:

1. **Automatic Event Detection:** Set ``config.options.detection = True`` to automatically detect events
2. **Pre-Provided Locations:** Set ``config.options.detection = False`` and provide ``config.detection.locs`` with known event times
3. **Custom Stage Execution:** Execute stages individually for fine-grained control
4. **Optional Stages:** Enable/disable stages (BIC selection, artifact removal, bootstrap analysis) based on needs

Implementation Details
=======================

Event Detection Algorithm
-------------------------

The event detection uses a robust threshold-based approach:

1. **Threshold Calculation:**
   
   .. code-block:: python

      threshold = np.nanmean(detection_signal) + thres_ratio * np.nanstd(detection_signal)

2. **Initial Detection:**
   
   .. code-block:: python

      temp_locs = np.where(detection_signal >= threshold)[0]

3. **Peak Alignment (if selected):**
   
   - For each detected location, finds the local peak within a window of size ``l_extract``
   - Uses ``find_peak_locations()`` utility function
   - Ensures events are aligned to maximum amplitude

4. **Pooled Alignment (if selected):**
   
   - Uses detected locations directly
   - Optional shrinking: reduces redundant detections when events are temporally close
   - Uses ``shrink_locations_resample_uniform()`` and ``find_best_shrinked_locations()`` utilities

Snapshot Extraction Details
----------------------------

The snapshot extraction creates a 3D array suitable for VAR model estimation:

- **Shape:** ``(n_variables × (model_order + 1), n_time_points, n_trials)``
- **Lagged Variables:** Includes ``model_order`` lags of each variable for VAR modeling
- **Time Alignment:** All events are aligned to the same temporal reference point
- **Boundary Handling:** Out-of-bounds windows are filled with NaN and logged

Error Handling
---------------

The pipeline includes comprehensive error handling:

- **Input Validation:** Each stage validates its inputs before processing
- **Graceful Degradation:** Missing optional parameters use sensible defaults
- **Detailed Logging:** All stages log their progress and any issues encountered
- **Exception Propagation:** Errors are caught, logged, and re-raised with context

Performance Considerations
----------------------------

- **Vectorized Operations:** Uses NumPy vectorization for efficient array operations
- **Memory Efficiency:** Processes data in-place where possible
- **Lazy Evaluation:** Optional stages (BIC, bootstrap) are only executed when enabled
- **Parallelization Ready:** Stage-based design allows for future parallel execution of independent stages

Integration with Causality Analysis
====================================

The preprocessing pipeline is tightly integrated with TranCIT's causality analysis methods:

1. **DCS/rDCS:** The extracted event snapshots and computed statistics are directly used by ``DCSCalculator`` and ``RelativeDCSCalculator``
2. **Transfer Entropy:** Event-aligned data enables time-varying TE computation
3. **Granger Causality:** VAR model statistics from preprocessing are used for GC computation

The pipeline output (``PipelineResult``) contains all necessary data structures for immediate causality analysis without additional preprocessing.

References
===========

For detailed API documentation, see the :doc:`API Reference <api>` (specifically the :ref:`pipeline-system` section).

For usage example, see:

- `CA3-CA1 Causality Example <https://github.com/CMC-lab/TranCIT/blob/main/examples/compute_ca3_ca1_causality.py>`_

