# SMPS: Soil Water Potential Prediction System

A physics-ML hybrid system for irrigation optimization using soil water potential (ψ) as the fundamental variable.

## 🔬 Scientific Foundation

SMPS addresses critical failures in traditional soil moisture-based irrigation systems:

- **ψ-Space Universality**: Uses soil water potential as the fundamental variable
- **Physics-ML Hybrid**: Combines rigorous soil physics with machine learning
- **Ensemble Uncertainty**: Quantifies prediction uncertainty for risk-aware decisions
- **Scientific Validation**: Rigorous hypothesis testing and evidence-based methods

## 🛠️ Installation

### Option 1: Conda Environment (Recommended)

```bash
# Create reproducible environment
conda env create -f environment.yml
conda activate smps

# Install SMPS
pip install -e .[dev,ml,data,reproducibility]
```

### Option 2: Pip Installation

```bash
# Install with all dependencies
pip install -e .[dev,ml,data,reproducibility]
```

## 🔒 Reproducibility

SMPS implements comprehensive reproducibility controls to ensure scientific results are completely reproducible:

### Master Seed Management

All stochastic processes derive from a single master seed (default: 42):

```python
from smps.core.reproducibility import initialize_reproducibility

# Initialize with master seed
initialize_reproducibility(master_seed=42)
```

### Environment Specification

Dependencies are pinned to exact versions in `pyproject.toml`:

```toml
numpy==1.24.3
pandas==2.0.3
scipy==1.11.3
scikit-learn==1.3.0
lightgbm==4.0.0
```

### Data Versioning

Complete data provenance tracking using DVC:

```python
from smps.data.versioning import track_dataset

# Track dataset with full provenance
provenance = track_dataset(
    df=my_dataframe,
    dataset_name="processed_soil_data",
    source_files=["data/raw/soil_data.csv"],
    transformation_script="scripts/preprocess_soil_data.py"
)
```

### Deterministic Execution Order

Operations execute in dependency order regardless of call sequence:

```python
from smps.core.execution_order import execute_pipeline

# Execute all tasks in correct order
results = execute_pipeline()
```

### Quick Reproducibility Setup

```bash
# Initialize complete reproducibility environment
make reproducibility

# Or manually:
python scripts/initialize_reproducibility.py
```

## 📊 Usage

### Basic Pipeline

```python
from smps.pipeline import SMPSPipeline

# Initialize with reproducibility
pipeline = SMPSPipeline(master_seed=42)

# Run complete pipeline
results = pipeline.run()

# Results include uncertainty quantification
print(f"Prediction: {results['prediction']:.3f} ± {results['uncertainty']:.3f}")
```

### Scientific Validation

```python
from scripts.scientific_validation import ScientificValidator

# Run comprehensive scientific validation
validator = ScientificValidator()
results = validator.run_complete_scientific_validation()

# Check if ψ universality hypothesis is rejected
if results['psi_universality']['hypothesis_rejected']:
    print("Evidence against ψ universality - soil-specific calibration required")
```

## 🧪 Testing

### Reproducibility Tests

```bash
# Run reproducibility-focused tests
pytest tests/ -k reproducibility -v

# Test that results are identical across runs
python scripts/test_reproducibility.py
```

### Physics Validation Tests

```bash
# Run physics pass/fail tests
pytest tests/ -k physics_pass -v
```

## 📁 Project Structure

```
smps/
├── src/smps/
│   ├── core/
│   │   ├── reproducibility.py      # Random seed management
│   │   └── execution_order.py      # Deterministic execution
│   ├── data/
│   │   └── versioning.py           # Data provenance tracking
│   ├── physics/
│   │   └── soil_physics_validator.py # Scientific validation
│   └── ml/
│       └── training.py             # Reproducible training
├── scripts/
│   ├── scientific_validation.py    # Validation pipeline
│   └── initialize_reproducibility.py # Setup script
├── data/
│   └── provenance/                 # Data version records
├── results/                        # Versioned results
└── environment.yml                 # Pinned conda environment
```

## 🔬 Scientific Validation Results

SMPS has been validated against fundamental soil science principles:

- **ψ Universality Test**: Strong evidence against universal ψ thresholds
- **PTF Ensemble Benefit**: Ensemble methods provide better uncertainty quantification
- **Water Balance Sensitivity**: Identified critical parameters requiring calibration
- **Cross-validation Robustness**: Site-blocked CV prevents geographic fingerprinting

## 📈 Key Features

- **Physics-First Approach**: Soil physics equations drive all predictions
- **Uncertainty Quantification**: Ensemble methods with confidence intervals
- **Site-Specific Calibration**: Accounts for soil texture effects
- **Temporal Dependencies**: Handles sequential irrigation decisions
- **Production Ready**: Containerized deployment with monitoring

## 🤝 Contributing

1. **Reproducibility First**: All changes must maintain reproducibility
2. **Scientific Rigor**: New features require scientific validation
3. **Testing**: Add tests for new functionality
4. **Documentation**: Update reproducibility documentation

### Development Setup

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
make test

# Check reproducibility
make reproducibility
```

## 📄 License

This project implements scientifically validated soil physics methods for irrigation optimization.

## 🔗 References

- Van Genuchten, M. Th. (1980). A closed-form equation for predicting the hydraulic conductivity of unsaturated soils. Soil Science Society of America Journal, 44(5), 892-898.

- Saxton, K. E., & Rawls, W. J. (2006). Soil water characteristic estimates by texture and organic matter for hydrologic solutions. Soil Science Society of America Journal, 70(5), 1569-1578.

- Vereecken, H., et al. (2010). On the spatio-temporal dynamics of soil moisture at the field scale. Journal of Hydrology, 385(1-4), 1-4.