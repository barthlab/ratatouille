# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Philosophy

Be brutally honest, don't be a yes man. 
If I am wrong, point it out bluntly. 
I need honest feedback on my code.

## Repository Overview

Ratatouille is a neuroscience data analysis framework for hierarchical calcium imaging, electrophysiological recordings, and behavioral data processing. The system handles multi-modal experimental data including calcium imaging (from Suite2p), electrophysiological recordings, behavioral tracking (licking, locomotion, pupil, whisker), and timeline synchronization.

## Project Architecture

### Core Philosophy
Data flows through a strict hierarchy: **Cohort → Mice → FOV → Session → Cell → Trial**, where each level enforces appropriate temporal and spatial constraints. This hierarchical structure mirrors experimental organization and ensures data integrity.

### Directory Structure
- `kitchen/` - Core analysis framework (the main package)
- `cook/` - Analysis pipelines and experimental scripts 
- `ingredients/` - Raw experimental data storage
- `cuisine/` - Analysis outputs, reports, and figures
- `critic/` - Comprehensive test suite

### Key Modules
- `kitchen/structure/` - Hierarchical data structures with coordinate validation
- `kitchen/loader/` - Data loading pipeline integrating Suite2p, electrophysiology, behavior, and timeline data
- `kitchen/plotter/` - Four-layer plotting system (macros → decorators → ax_plotter → unit_plotter)
- `kitchen/video/` - Video processing and behavioral feature extraction (including TIFF stack conversion)
- `kitchen/configs/` - Path routing and configuration management

## Development Commands

### Installation
```bash
pip install -e .
```

### Testing
```bash
# Manual testing by running analysis scripts
python cook/cook_template/cook_template.py
python cook/JS_juxtacelluar_analysis/overview.py
```

### Code Quality
```bash
# Format code
black kitchen/ critic/

# Type checking
mypy kitchen/

# Linting
flake8 kitchen/
```

### Running Analysis
```bash
# Execute analysis pipelines from cook/ directory
python cook/cook_template/cook_template.py
python cook/head_fixed_training/overview.py
python cook/JS_juxtacelluar_analysis/overview.py  # New: Juxtacellular analysis
```

## Common Development Patterns

### Data Loading
Two main entry points for different experiment types:
```python
# For calcium imaging experiments
import kitchen.loader.two_photon_loader as hier_loader
dataset = hier_loader.cohort_loader(
    template_id="RandPuff", 
    cohort_id="HighFreqImaging_202507"
)

# For electrophysiology experiments  
import kitchen.loader.ephys_loader as ephys_loader
dataset = ephys_loader.cohort_loader(
    template_id="PassivePuff_JuxtaCelluar_FromJS",
    cohort_id="SST_EXAMPLE"
)

# Quick diagnostic loading (FOV level only for calcium imaging)
diagnostic_data = hier_loader.naive_loader(
    template_id="RandPuff",
    cohort_id="HighFreqImaging_202507"
)
```

### Data Access Patterns
```python
# Access different hierarchy levels
fov_nodes = dataset.select("fov")
session_nodes = dataset.select("session") 
trial_nodes = dataset.select("trial")

# Generate status reports
dataset.status(save_path="status_report.xlsx")
```

### Plotting
```python
from kitchen.plotter.macros.basic_macros import session_overview, fov_trial_avg_default
from kitchen.plotter.plotting_manual import PlotManual

# Configure plotting modalities
plot_manual = PlotManual(
    timeline=True,
    fluorescence=True, 
    lick=True,
    locomotion=True,
    pupil=True,
    whisker=True
)

# Generate visualizations
for session in dataset.select("session"):
    session_overview(session, plot_manual=plot_manual)
```

## File Organization Patterns

### Data File Structure
Raw data in `ingredients/` follows this hierarchy:
```
ingredients/{cohort_id}/{mice_id}/{fov_id}/
├── soma/Fall.mat              # Suite2p calcium imaging output
├── timeline/TIMELINE_*.csv    # Experimental event timelines
├── lick/LICK_*.csv           # Lick behavior timestamps
├── locomotion/LOCOMOTION_*.csv # Rotary encoder data
├── video/VIDEO_*.avi         # Behavioral videos
└── pupil/PUPIL_*.csv         # Video-extracted pupil data
```

### Analysis Output Structure
Results in `cuisine/` mirror the input hierarchy for easy navigation and organization.

## Data Integration Details

The framework integrates four primary data streams:
1. **Fluorescence Data**: Suite2p/Fall.mat files with TTL synchronization
2. **Electrophysiological Data**: Pickle files with potential recordings, spike detection, and filtering
3. **Behavioral Data**: Multi-modal CSV files (lick, locomotion, pupil, whisker)
4. **Timeline Events**: Arduino/timeline files with stimulus timestamps

TTL-based synchronization aligns all modalities to a common timeline, with fallback to fixed-offset alignment when TTL unavailable.

## Key Classes

- `DataSet`: Manages collections of hierarchical nodes with fast lookup
- `Node` hierarchy: Base class with specialized subclasses for each level
- `NeuralData`: Central container integrating all experimental modalities (fluorescence, potential, behavior, timeline)
- `Potential`: Electrophysiological recordings with spike detection, filtering, and component analysis
- `PlotManual`: Configuration object controlling visualization modalities

## Configuration Files

Key settings in `kitchen/settings/`:
- `loaders.py` - Data file search modes (strict vs hodgepodge)
- `timeline.py` - Supported experimental events and alignment
- `trials.py` - Trial extraction windows and alignment parameters
- `behavior.py` - Behavioral data processing parameters
- `fluorescence.py` - Calcium imaging analysis settings
- `potential.py` - Electrophysiology recording parameters (spike detection, filtering, thresholds)

## Development Strategy

Development and validation is done through:
- Running analysis scripts in `cook/` directory to test functionality
- Manual inspection of data loading and plotting outputs
- Visual verification of results in `cuisine/` output directory

Use the analysis scripts to validate changes and ensure data processing works correctly.