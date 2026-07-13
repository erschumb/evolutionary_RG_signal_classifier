# Purifying selection in the human population identifies functional RG motifs and predicts novel RNA-binding proteins



This repository contains code and data used in a submitted publication. This will be updated and link to the publication once it is accepted.



## Repository structure

```

|── data/                # Data stored necessary for this project

│ ├── external            # External data

│ ├── processed           # Processed files necessary for the statistical analysis

│ ├── results             # Output files and figures

│ ├── external            # External data

│ ├── processed           # Processed files necessary for the statistical analysis

│ ├── results             # Output files and figures

├── src        # Subfigures necessary to recreate the excat figures from the publication

├── figures        # Subfigures necessary to recreate the excat figures from the publication

├── models        # Subfigures necessary to recreate the excat figures from the publication

├── 0\_preprocessing.ipynb     # Code for processing the raw data and generating the necessary datasets

├── 1\_gnomAD\_processing.ipynb             # Script for statistical analysis of phys/chem properties of RG motifs and general properties (length, impurity, etc.)

├── 2\_variant\_analysis\_visualization.ipynb                 # Scripts for statistical analysis of IDR-related properties

├── 3\_RF\_model\_creation.ipynb              # Scripts for statistical analysis of domain-focused analysis

├── 4\_inference\_rg\_motifs.ipynb                  # Scripts for statistical analysis of detailed amino acid composition

├── LICENSE               # License file

├── environment.yml       # Necessary dependencies for project

└── README.md             # This file

```

## Data Flow



The following diagram shows the data acquisition, processing and organization required for the project and required as prerequisites for running the analysis scripts.

This summarizes the work done in '0\_preprocessing.ipynb' and '1\_gnomAD\_processing.ipynb'.



!\[Dataflow Diagram](docs/data_flow.png)



## Requirements and Installation



All dependencies for this project are listed in the provided `environment.yml` file.



### Missing file (due to file size)



The `human\_reviewed.json` from PhasePred containing information on the phase separation propensity of all human proteins has not been included in this repo, due to its large file size.

The most recent version can be downloaded here: http://predict.phasep.pro/static/phasepred/database/human\_reviewed.zip

The version used for this was the file from Feb, 11th, 2022, downloaded on June, 13th, 2023 and can be requested if necessary, see Contact.

At the time of writing (July, 1st, 2025) this is still the latest release.



### Recommended (Using Conda)



Create the environment:



```bash conda env create -f environment.yml ```



### Activate the environment:



```bash conda activate your-environment-name ```



## How to use:



To recreate this analysis, the preprocessing `0\_preprocessing\_of\_proteome.ipynb` is necessary to create/clean/annotate the necessary datasets. The statistical analyses (1-4) can be run in independent order and edited/expanded independently from each other.



## License:



see `LICENSE` file



## Contact



For questions or suggestions, please contact:

Eric Schumbera

e.schumbera@uni-mainz.de



