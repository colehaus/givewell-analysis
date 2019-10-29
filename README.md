This is a Python analysis which implements [GiveWell's charity cost-effectiveness models](https://www.givewell.org/how-we-work/our-criteria/cost-effectiveness/cost-effectiveness-models/changelog-2019#Version_4_Published_May_29_2019) in order to perform uncertainty and sensitivity analysis. Accompany prose can be found at [Collectively Exhaustive](https://www.col-ex.org/series/GiveWell%2520cost-effectiveness%2520analysis%2520analysis/).

The core libraries relied upon are [PyMC3](https://docs.pymc.io/) and [SALib](https://salib.readthedocs.io/en/latest/).

If you already have the package manager [Nix](https://nixos.org/nix/), you should be able to get up and running by:

```sh
git clone https://github.com/colehaus/givewell-analysis
cd givewell-analysis
nix-shell
python -c "from test import test; test()"
```

The overall project structure is as follows:

- Each model (roughly corresponding to each tab in GiveWell's spreadsheet) gets its own file. For example, `cash.py` and `nets.py`.
- Each such model consists of pure functions which break the model into pieces. The model files also contain model-specific parameters (e.g. "net use adjustment: 0.90") pulled from the spreadsheets.
- The `shared_params.py` file contains parameters which are shared across multiple models.
- The `models.py` file pulls all the parameters into a single dictionary, pulls all the models into a single dictionary, and specifies how to weave together the models with the parameters.
- `tree.py` contains helpers for working with the models when regarded as trees of function calls.
- `main.py` performs all the analysis.
- `test.py` is a simple sanity check that the models as implemented reproduce GiveWell's results.
