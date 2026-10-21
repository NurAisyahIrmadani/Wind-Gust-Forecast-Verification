# Wind-Gust-Forecast-Verification
Python and MATLAB scripts for short-range wind and wind gust forecast verification, time-series visualization, Taylor diagrams, and empirical quantile mapping (EQM) bias correction.

This repository supports the research study "Improving Short-Range Wind and Wind Gust Prediction for Artisanal Fisherman Safety at Paotere Port, Makassar, Indonesia."

## Repository Structure
```text
Wind-Gust-Forecast-Verification/
├── README.md
├── LICENSE
├── .gitignore
├── python/
│   ├── README.md
│   ├── 03_timeseries_wind_gust_combined.py
│   ├── 04_skill_wind_gust_combined.py
│   └── 05_eqm_postprocessing_combined.py
└── matlab/
    ├── README.md
    ├── Wind_TD2.m
    └── WindGust_TD2.m
```

## Research Scope
The scripts support the analysis of short-range wind speed and wind gust forecasts, including:
1. Time-series visualization
2. Statistical forecast verification
3. Taylor diagram visualization
4. Empirical quantile mapping (EQM) bias correction
5. Leave-one-day-out cross-validation (LODO-CV), where implemented

## Scripts
## Python Scripts

| Script | Description |
|---|---|
| `03_timeseries_wind_gust_combined.py` | Visualizes wind speed and wind gust time series. |
| `04_skill_wind_gust_combined.py` | Evaluates forecast performance using statistical metrics. |
| `05_eqm_postprocessing_combined.py` | Performs EQM bias correction and LODO-CV evaluation. |

## MATLAB Scripts

| Script | Description |
|---|---|
| `Wind_TD2.m` | Generates a Taylor diagram for wind speed. |
| `WindGust_TD2.m` | Generates a Taylor diagram for wind gust. |

## Data Sources
ERA5 reanalysis data are available through the Copernicus Climate Change Service (C3S) Climate Data Store:

https://cds.climate.copernicus.eu/

Wind forecast information was collected manually from the Windy platform using its ECMWF-based forecast display.

Raw input datasets are not included in this repository. Users should obtain data from the relevant sources and comply with applicable access, licensing, and redistribution conditions.

## Requirements
Python scripts require Python and the scientific computing libraries used in each script. MATLAB scripts require MATLAB and any external toolbox specified in the code.

Input filenames, worksheet names, column names, date-time formats, units, and forecast lead times may need to be configured before running the scripts.

## Reproducibility
These scripts document and support the analyses described in the associated research study. Reproducing the reported results requires compatible input data, consistent processing settings, and the appropriate software dependencies.

Users should verify the script configuration and numerical outputs before comparing results with those reported in the manuscript.

## License
The code in this repository is distributed under the MIT License, subject to the rights of the respective copyright holders. The MIT License does not grant additional rights to third-party forecast or reanalysis datasets.

## Citation
Please cite the associated research study when its publication details become available.

## Contact
For questions regarding this research, please contact:

- **Nur Aisyah Irmadani** — Research and code contributor
- **Prof. Dr. Halmar Halide, M.Sc.** — Research supervisor and corresponding author

Email addresses:
- Nur Aisyah Irmadani: aisyahirmadani21@gmail.com
