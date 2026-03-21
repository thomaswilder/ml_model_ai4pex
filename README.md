# A Neural Network for the prediction of Eddy Kinetic Energy in GEOMETRIC

CNN and U-Net models for a Machine Learning eddy parameterisation for [NEMO](https://www.nemo-ocean.eu/) developed as part of the [AI4PEX](https://ai4pex.eu/) project. The models learn to predict eddy kinetic energy from eddy-permitting resolution variables. The NN will form an ML-GEOMETRIC to replace the eddy energy budget in [Mak et al., 2023](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS003886).

## Installation

Clone the repository and install in editable mode into your Python environment:

```bash
git clone https://github.com/<your-username>/ml_model_ai4pex.git
cd ml_model_ai4pex
pip install -e .
```

This installs the `ml_model_ai4pex` package and all required dependencies.

## Dependencies

- Python >= 3.9
- TensorFlow / Keras
- Tensorboard
- xarray[io]
- xbatcher
- numpy
- PyYAML

## Usage

Copy and edit the example configuration file, then run the model script:

```bash
cp run/config_model.yml my_config.yml
# Edit my_config.yml to point to your data files and set hyperparameters
python run/run_model.py --config my_config.yml
```

You can also override options at runtime
```bash
python run/run_model.py --config my_config.yml --epochs=200
```

### Configuration

`run/config_model.yml` controls all aspects of the run:

| Key | Description |
|-----|-------------|
| `model` | Architecture to use: `unet` or `cnn` |
| `mode` | `train` or `predict` |
| `data_filenames` | NetCDF file with inputs and target |
| `model_dir` | Directory for saving model |
| `epochs` | Number of training epochs |
| `batch_size` | Batch size |
| `learning_rate` | Initial learning rate |

Do `python run_model.py -h` for more available options including architecture parameters, normalization statistics, and early stopping settings.

## Project Structure

```
ml_model_ai4pex/
├── ml_model_ai4pex/                    # Installable Python package
│   ├── __init__.py
│   ├── cnn.py                          # Fully convolutional architecture
│   ├── unet.py                         # U-Net encoder-decoder architecture
│   ├── model_components.py             # Shared Keras layers and loss functions
│   ├── model_setup.py                  # Scenario and data setup
│   ├── preprocess_data.py              # Data loading and preprocessing
│   ├── train_model.py                  # Training loop
│   ├── predict_model.py                # Inference pipeline
│   └── parsing_args.py                 # CLI argument parser
├── run/
│   ├── run_model.py                    # Main entry point script
│   ├── config_model.yml                # Example configuration file
│   └── submit_test.sh                  # Example SLURM submission script
├── examples/
│   ├── sample_dataset_for_github/
│   │   └── sample*                     # Sample training datasets including input and target
│   ├── sample_model/
│   │   └── sample*                     # Sample pre-trained keras model
│   └── sample_offline_analysis.ipynb   # Sample notebook
└── pyproject.toml
```

## Running on a Cluster (SLURM)

An example SLURM submission script is provided at `run/submit_model.sh`. Edit the paths and resource requirements to match your cluster configuration before submitting. Currently spec'd for [JASMIN](https://jasmin.ac.uk/).

## Running example notebook

A sample testing dataset and simple pre-trained keras model are found in directories `examples/sample_*`.

You can run this notebook yourself by installing `ml_model_ai4pex` and installing the following pip packages: `matplotlib`, `cmocean`, `skimage` and `sklearn`.

## Data & Reproducibility

### Data

The U-Net model is trained on high-resolution data from the [DINO configuration ](https://github.com/vopikamm/DINO/tree/DINO_4.2.1). The model uses coarse-grained fields of: $\zeta/f$ (vorticity over Coriolis parameter), coarse kinetic energy, $L_d/\Delta s$ (baroclinic deformation radius over grid scale), to predict fine kinetic energy, or eddy kinetic energy.

The full dataset will be uploaded to a Zenodo repository on publication.

### Apply your own data

You can of course do this. Our workflow uses a combination of [xnemogcm](https://github.com/rcaneill/xnemogcm), [CDFTOOLS](https://github.com/meom-group/CDFTOOLS/tree/master), [xesmf](https://github.com/pangeo-data/xESMF), and [gcm-filters](https://github.com/ocean-eddy-cpt/gcm-filters).
.

## License

The code in this repository is licensed under the MIT License. See the `LICENSE` file for details. The full dataset used for the project is not included and is subject to separate access and use restrictions.

