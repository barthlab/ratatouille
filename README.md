# Ratatouille

A neuroscience data analysis framework for hierarchical calcium imaging and behavioral data processing.

## Features

• **Hierarchical Data Organization** - Structured data management from cohort level down to individual cells and trials

• **Multi-Modal Data Integration** - Seamlessly combines calcium imaging, behavioral tracking, and timeline data

• **Automated Video Processing** - Built-in tools for optical flow analysis, pupil tracking, and whisker detection

• **Flexible Data Loading** - Template-based loaders for different experimental protocols

• **Comprehensive Visualization** - Plotting utilities for neural traces, behavioral data, and synchronized views

## Quick Start

### Installation

```bash
pip install -e .
```

### Requirements

- Python ≥3.11
- NumPy, Pandas, SciPy, OpenCV
- See `pyproject.toml` for complete dependencies

### Basic Usage

```python
import kitchen.loader.hierarchical_loader as hier_loader

# Load experimental data
dataset = hier_loader.cohort_loader(
    template_id="RandPuff", 
    cohort_id="HighFreqImaging_202507"
)

# Generate status report
dataset.status(save_path="status_report.xlsx")
```

## Project Structure

```
├── kitchen/          # Core analysis framework
│   ├── loader/       # Data loading utilities
│   ├── structure/    # Hierarchical data structures
│   ├── plotter/      # Visualization tools
│   ├── video/        # Video processing modules
│   ├── operator/     # Data manipulation operators
│   ├── configs/      # Configuration and routing
│   ├── settings/     # Module-specific settings
│   ├── utils/        # Utility functions
│   └── writer/       # Data export utilities
├── cook/             # Analysis pipelines and experiments
├── ingredients/      # Raw experimental data
├── cuisine/          # Analysis outputs and reports
└── critic/           # Test suite
```

## License

MIT License - see LICENSE file for details.
