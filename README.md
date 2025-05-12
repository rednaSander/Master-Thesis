# Master-Thesis
## ModifAInder Guide

A comprehensive tool for generating in-silico spectral libraries that integrates modifications with deep learning predictions to create data driven spectral libraries for candidate modification site discovery in DIA mass spectrometry experiments.
This tool uses a modified version of PhosphoLingo as the modification prediction engine, which has been adapted to prediction only on sites that occur on peptides that fall whithin library configurations. 

For more details on PhosphoLingo, including training different models, visit the PhosphoLingo GitHub repository:
(https://github.com/jasperzuallaert/PhosphoLingo)

### Overview

This pipeline consists of two main steps:

1. **PhosphoLingo Prediction**: Predicts modification sites on peptides using trained models 
2. **ModifAInder**: Generates spectral libraries incorporating PhosphoLingo predictions along with fixed and variable modifications

### Features

- In-silico protein digestion with configurable parameters
- Integration of PhosphoLingo predictions for targeted modifications
- Support for variable and fixed modifications
- Customizable charge state assignments
- Theoretical spectra generation using MS²PIP
- Retention time predictions using DeepLC
- Optional ion mobility predictions with IM2Deep
- Centralized configuration system

### Installation

```bash
# Clone the repository
git clone https://github.com/rednaSander/Master-Thesis
cd ModifAInder

# Install dependencies (for each of the dependency files)
pip install -r <dependencies.txt>
```

### Usage

The workflow requires running two commands sequentially:

1. First, run PhosphoLingo to generate modification predictions:
```bash
python phospholingo predict <path to library config>
```

2. Then, run ModifAInder to generate the spectral library:
```bash
python ModifAInder.py --config <path to library config>
```

### Configuration

The configuration file (JSON format) controls both steps of the pipeline. Here's an example structure:

```json
{
    "proteome_fastas": ["path/to/proteome.fasta"],
    "missed_cleavages": 1,
    "min_length": 7,
    "max_length": 30,
    "control_library": false,
    "predictions": [
        {
            "pred_csv": "output/predictions.csv",
            "PTM": "Y[Phospho]",
            "threshold": 0.7,
            "model": "models/phospho_model.pt"
        }
    ],
    "modifications": [
        {
            "label": "Oxidation",
            "amino_acid": "M",
            "fixed": false
        },
        {
            "label": "Carbamidomethyl",
            "amino_acid": "C",
            "fixed": true
        }
    ],
    "max_variable_mods": 3,
    "charges": [1, 2, 3, 4],
    "add_retention_time": true,
    "add_ion_mobility": true,
    "mz_precursor_range": [300, 1800],
    "frag_mass_lower": 200.0,
    "frag_mass_upper": 1800.0,
    "ms2pip_model": "HCD",
    "batch_size": 100000,
    "processes": 10,
    "out_path": "output/",
    "out_name": "speclib"
}
```

#### Configuration Parameters

The configuration files of each spectral library that was used in this thesis can be found under "library_configs" of this repo.    

##### General Parameters
- `proteome_fastas`: List of FASTA files containing protein sequences
- `peptides`: Whether to use peptide-based (true) or full protein-based (false) prediction
- `missed_cleavages`: Number of allowed missed cleavages
- `min_length`, `max_length`: Peptide length constraints
- `control_library`: Whether to generate a control library by random sampling

##### PhosphoLingo Parameters
- `predictions`: List of modification predictions to generate
  - `ms2pip_model`: Path to trained PhosphoLingo model
  - `pred_csv`: Output file for predictions
  - `PTM`: Modification type and residue (e.g., "Y[Phospho]" or "ST[Phospho]")
  - `threshold`: Confidence threshold for predictions (0.0-1.0)

##### Modification Parameters
- `modifications`: List of modifications to consider
  - `label`: Modification name
  - `amino_acid`: Target residue
  - `fixed`: Boolean indicating if modification is fixed
- `max_variable_mods`: Maximum variable modifications per peptide

##### Spectral Library Parameters
- `charges`: Charge states to consider
- `add_retention_time`: Enable retention time prediction
- `add_ion_mobility`: Enable ion mobility prediction
- `mz_precursor_range`: Mass range for precursors [min, max]
- `ms2pip_model`: MS2 prediction model type
- `batch_size`: Processing batch size
- `processes`: Number of parallel processes
- `frag_mass_lower`: Lower mass limit for fragment ions
- `frag_mass_upper`: Upper mass limit for fragment ions
- `out_path`: Output directory
- `out_name`: Output filename

## Output

The pipeline generates:
1. CSV files containing modification predictions from PhosphoLingo
2. A spectral library file (.msp format) containing:
   - Peptide sequences with modifications
   - Theoretical MS2 spectra
   - Predicted retention times
   - Predicted collision cross-sections (if enabled)

## Results
All raw search engine data and associated files are available via Zenodo: 

https://zenodo.org/records/15007519

The data is organized as follows:
```
results.zip
├── Orbitrap AIF
│   ├── diann
│   │   └── raw search engine outputs
│   └── alphadia
│       └── raw search engine outputs 
├── TTOF6600 SWATH
│   ├── diann
│   │   └── raw search engine outputs
│   └── alphadia
│       └── raw search engine outputs
├── TimsTOF Pasef
│   └── diann
│       └── raw search engine outputs (with XICs)
│       
└── Orbitrap Exploris 480
    ├── diann
    │   └── raw search engine outputs
    └── alphadia
        └── raw search engine outputs
```
Data from other sources such as UniProt, PhosphoSitePlus, PhosphoLingo Training data, PhosphoLingo Predictions can be found in data.zip:
```
data.zip
├── proteome fasta files
├── phospholingo models training data
├── phospholingo predictions used to make the spectral libraries
└── phosphosite-plus literature data
```
### Data Analysis
All data analysis scripts (jupyter notebook files) can be found in the "data analysis" folder of this repo. Here is an overview of each file: 

| Section   | Notebooks | Description |
|-----------|-----------|-------------|
| 5.1, 5.2 | section_5.1_5.2, section_5.1, section_5.1_clustering_analysis | Method Validation and Reduction in Computational Resources, cluster analysis: figure 8, Venn diagram: figure 9 |
| 5.3       | section_5.3_DIANN.ipynb, section_5.3_AlphaDIA.ipynb | Analysis of regular sample datasets |
| 5.4, 5.6 | section_5.4_DIANN_5.6, section_5.4_AlphaDIA.ipynb | Analysis of enriched sample datasets, Comparing performance of different predictive models |
| 5.5, 5.7 | section_5.5, section_5.7 | Modification localisation scores, Impact of filtering on novel phosphopeptide discovery |
| 5.8       | section_5.8_DIANN, section_5.8_AlphaDIA  | Literature validation of identified modification sites |

*Figure 18 from section 5.5 was created using DIA-NN's built-in viewer with the XICs from the timsTOF PASEF analyses (available via Zenodo).
