# srlab: reference code for "Regularized free deconvolution for spectral repair of resampling distortion in high-dimensional particle filters"

This repository contains the reference implementation and experiment scripts behind the paper. The paper and its supplementary material are complete on their own; the code is provided as a reference for checking the reported numbers, not as a maintained library.

## Contents

| Path | Purpose |
|---|---|
| `srlab/` | The package: forward spectral map and its fixed-point solver (`solver.py`, `forward.py`), count laws (`laws.py`), regularized inversion (`inverse.py`), supervisor (`supervisor.py`), GPU kernel and precision ladder (`gpu.py`, `kernels/`, `tail.py`) |
| `srlab/experiments/` | One module per experiment: `exp1` (static distortion and repair), `exp2` (certificates and identifiability), `exp3` (full cycle), `tableI` (Metropolis preconditioning), `s7` (activation zone), `dfb_n` (fallback error against N), `exp1_lw` (Ledoit–Wolf baseline) |
| `scripts/` | Figure and table scripts |
| `tests/` | Core tests (solver, count laws, inversion, and equivalence with the frozen reference) |
| `reference/` | Frozen earlier implementation, used only as the oracle in the equivalence tests |
| `manifests/` | The registered run of each experiment: command, environment, and the sha256 of every output |
| `PREREGISTRATION_SUMMARY.md` | How the experimental protocol was fixed and how it changed |

## Requirements

Python 3.13, `numpy==2.4.5`, `scipy==1.17.1`, `matplotlib` for the figures. A CUDA build of PyTorch is optional: the forward solver uses the GPU when available and otherwise falls back to the CPU, recording the path it took. The paper's numbers were produced on the GPU path; a CPU run reproduces them within the declared backend tolerance (integrated quantities to 10⁻¹³), not necessarily bitwise.

## Usage

```
pip install numpy==2.4.5 scipy==1.17.1 matplotlib
python tools/run_tests.py                 # core tests, a few seconds on a CPU
python -m srlab list                      # registered experiments and figure steps
python -m srlab run exp1                  # one experiment, written to results/rerun/
python -m srlab verify                    # compare a re-run with the registered sha256
```

Figures are drawn from the data package archived on Zenodo (DOI [to be filled]). Unpack it and set `SRLAB_DATA` to its location before `python -m srlab run all`.

Set `OMP_NUM_THREADS=1` (and the MKL and OpenBLAS equivalents) and `PYTHONHASHSEED=0` for exact reproduction; the thread count is part of the numerical configuration, and Table 1 pins it to 8 internally.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

Code: MIT (`LICENSE`). Data: see the Zenodo record.
