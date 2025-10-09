# LS-OGD: Lyapunov Stable Adaptive Control for Multimodal Concept Drift

This repository contains the code for the paper "Lyapunov-Stable Adaptive Control for Multimodal Concept Drift" (LS-OGD). LS-OGD is a novel adaptive control framework for robust multimodal learning in non-stationary environments where concept drift can degrade performance. The system employs an online controller that dynamically adjusts the model's learning rate and the fusion weights between different data modalities in response to detected drift and evolving prediction errors. Theoretical guarantees establish that under bounded drift, the LS-OGD system's prediction error is uniformly ultimately bounded and converges to zero if the drift ceases.

<img width="3935" height="2901" alt="lsogd" src="https://github.com/user-attachments/assets/fe6dbb0b-0f98-4563-b7c8-1aba1e98695c" />


## Core Idea

The core of LS-OGD lies in integrating control-theoretic principles, specifically Lyapunov stability, into the multimodal learning process. This allows the system to:
* Continuously adapt to changes in data distribution (concept drift).
* Dynamically adjust the learning rate ($\eta_t$) for optimal adaptation speed.
* Adaptively re-weight the contribution of different modalities ($\alpha_t$) using a fusion mechanism, effectively mitigating modality-specific drifts.
* Maintain system stability and ensure bounded prediction error during drift.

## Repository Structure

The main components of this repository are organized as follows:

* `main_D.py`: The primary script to run the experiments. It handles data splitting, model initialization, training (Phase 1), concept drift simulation, adaptation (Phase 2 with LS-OGD controller), and evaluation against a static baseline.
* `model.py`: Defines the neural network architectures used, including:
    * `TextEncoder`: Encodes textual input (e.g., using CLIPTextModel[1]).
    * `ImageEncoder`: Encodes visual input (e.g., using CLIPVisionModel[1]).
    * `Fusion`: Implements the fusion strategy (e.g., `WeightedAverage`) to combine outputs from text and image encoders, which includes the adaptable fusion parameter $\alpha_t$.
* `data_utils.py`: Contains utilities for data loading and processing.
    * `LabeledDataset`: Loads labeled data and applies configured concept drifts (image degradation, text semantic shift) on-the-fly during Phase 2.
    * Drift functions like `image_drift_degradation` and `text_drift_semantic_shift`.
* `utility.py`: Provides helper functions for:
    * Logging setup.
    * Setting random seeds for reproducibility.
    * Calculating evaluation metrics (accuracy, F1-score, ECE, etc.).
    * Implementing controller actions (adjusting learning rate and fusion alpha).
    * Estimating drift signals.
* **Configuration Files (e.g., `D_42.yaml`)**: YAML files used to specify all experiment parameters, including dataset paths, model choices, drift settings, optimizer details, controller parameters, and training steps.

## Setup and Installation

1.  **Clone the repository.**
2.  **Python Environment:** A Python environment (e.g., Conda or venv) is recommended. The code is developed with Python 3.8+.
3.  **Install Dependencies:** Install the required Python packages. Key dependencies include:
    * `torch` (PyTorch)
    * `transformers` (Hugging Face Transformers)
    * `pandas`
    * `numpy`
    * `scikit-learn`
    * `matplotlib`
    * `PyYAML`

    You can typically install these using pip:
    ```bash
    pip install torch torchvision torchaudio
    pip install transformers pandas numpy scikit-learn matplotlib pyyaml
    ```
    (Note: Ensure PyTorch is installed according to your CUDA version if GPU support is needed.)

## Running the Experiments

1.  **Prepare the Dataset:**
    * The experiments are designed to use the M3A dataset.
    * Ensure the dataset (e.g., `combined_data.csv`) is placed in the directory specified by `data_root` and `dataset_folder_name` in your configuration file (e.g., `D_42.yaml`). The default is `data_root: "your_data_root_here"` and `dataset_folder_name: "M3A"`.

2.  **Configure Your Experiment:**
    * Modify an existing `.yaml` configuration file (like `D_42.yaml`) or create a new one to set your desired parameters for the dataset, model, drift, controller, and training.

3.  **Run the Main Script:**
    Execute the `main_D.py` script, providing the path to your configuration file:
    ```bash
    python main_D.py --config path/to/your_config.yaml
    ```
    For example, to run with the provided `D_42.yaml`:
    ```bash
    python main_D.py --config D_42.yaml
    ```

    The script will:
    * Load the data and split it into Phase 1 (initial training) and Phase 2 (drift/adaptation).
    * Train the initial model in Phase 1 (or load a pre-trained Phase 1 model if available and not forced to retrain).
    * In Phase 2:
        * Evaluate a static baseline model (Phase 1 without adaptation) on the drifted data.
        * Run the LS-OGD model, allowing it to adapt to the concept drift using the configured controller.
    * Log metrics and save results.

## Output

The script generates several outputs, saved in the directory specified by `save_path` and `run_name` in the configuration file:

* **Log files (`.log`):** Detailed logs of the experiment setup and progress.
* **Configuration file (`effective_config.yaml`):** A copy of the exact configuration used for the run.
* **Model Checkpoints (`.pt`):**
    * Master checkpoint from Phase 1 training (stored in `save_path/common_phase1_model_store/`).
    * Checkpoints during Phase 2 adaptation (if `save_every_phase2` is configured).
    * Final adapted model from Phase 2.
* **Plots (`.png`):** Visualizations of key metrics over time, comparing LS-OGD with the static baseline. This includes plots for:
    * Accuracy
    * F1 Score
    * Expected Calibration Error (ECE)
    * Adaptation Signals (Learning Rate and Fusion Alpha)
    * Error Signal and Estimated Drift Signal
    * Delta Error Signal (Lyapunov Proxy)
    * Cumulative Controller Cost

The code implements the LS-OGD framework and the experimental setup described in the submitted paper. Configuration files enable the reproduction of the reported experiments and the exploration of different settings.

## Reference
[1] Radford, Alec, et al. "Learning transferable visual models from natural language supervision." International conference on machine learning. PmLR, 2021.
