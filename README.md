# Anonymous submission to Language 2024 Fall "Towards A Two-Stage Phonotactic-Alternation Learning Model"


This guide will walk you through the process of setting up the required packages and dependencies to run this project.

## Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.7+**
- **Conda** (for package management)

## Installation Instructions

### 1. Download the zip file and decompress it to your local folder, then run the following on your terminal to locate the files (change to your own path)

```bash
cd /Users/.../2024-fall-language-submission
```


### 2. Install Python packages

Next, install the required packages. Some of the packages can be installed using `pip`, but for `pynini`, you will need to use Conda.

#### 2.1 Install `pynini` via Conda

`pynini` requires installation through Conda. Use the following command to install `pynini` from the `conda-forge` channel:

```bash
conda install -c conda-forge pynini
```

For more details about `pynini`, visit the [official documentation](https://www.openfst.org/twiki/bin/view/GRM/Pynini).

#### 2.2 Install other dependencies via `pip`

After `pynini` is installed, you can install the remaining dependencies using `pip`:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn researchpy statsmodels plotnine
```


## Troubleshooting

If you encounter any issues with installing `pynini`, ensure that:
- You're using a Conda environment.
- You're installing it via the `conda-forge` channel as shown above.

## Additional Resources

For more information about `pynini`, you can check out the [Pynini documentation](https://www.openfst.org/twiki/bin/view/GRM/Pynini).


## Run the code
For example, if you want to see how the two-stage learner works for finnish data, run:
```bash
python your_path/alternation_learner_finnish.py
```

Thank you!