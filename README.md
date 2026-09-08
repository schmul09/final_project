# Fair Value Estimation for Energy Derivatives with Neural Networks

A Hedging Application to the Russian Refined Products Shortage

## Overview

This project benchmarks a neural network against the Black-76 parametric model for fair value estimation of European-style vanilla options on ICE Brent Crude futures. The networks are trained to predict an empirically-derived local hedge ratio directly, rather than a fair-value price differentiated for Greeks. Two networks are trained separately — one for calls, one for puts — each using a sigmoid-bounded output matched to their respective delta ranges.

The dataset spans three time windows chosen around a geopolitical shock event (Ukrainian refinery strikes and the resulting Russian refined-products disruptions):

- **Prior shock window** — September to December 2025
- **Calm baseline window** — January to mid-February 2026
- **Test shock window** — May to June 2026

## Requirements

- Python 3.10+ (required by the `databento` client library)

## Repository Structure

```
final_project/
├── src/
│   ├── data_extraction.py             # Pulls raw option/futures data from Databento
│   ├── fix_futures_settle.py          # Re-sources futures settlement price via Statistics schema
│   ├── add_sofr.py                    # Merges in daily SOFR rate series (FRED)
│   ├── data_preparation.py            # Cleans and aligns raw data across windows
│   ├── inspect_arbitrage_violations.py# Manual inspection of no-arbitrage bound violations
│   ├── construct_label.py             # Builds the empirical hedge-ratio training label
│   ├── check_label_distribution.py    # Checks label distribution / outlier bounds
│   ├── multicollinearity_check.py     # VIF / correlation diagnostics on candidate features
│   ├── finalise_data.py               # Produces model-ready datasets
│   ├── train_model.py                 # Trains the calls and puts networks (with L2 weight decay)
│   ├── train_modelwd0.py              # Same, with weight decay disabled (ablation)
│   ├── evaluate_model.py              # Evaluates against Black-76 on the held-out test window
│   ├── evaluate_modelwd0.py           # Evaluates the wd0 ablation checkpoints
│   ├── check_generalisation.py        # Compares validation vs. test MSHE (generalisation gap)
│   ├── diagnose_regime_shift.py       # Out-of-distribution error analysis (all 4 features)
│   ├── diagnose_regime_no_rate.py     # Same, excluding rate from the OOD score
│   ├── run_ablation.py                # Multi-seed weight-decay ablation (with/without L2)
│   ├── weight_ablation.py             # Multi-seed reliability-weighting ablation
│   ├── diagnose_dates.py              # Checks date coverage across the three windows
│   └── generate_figures.py            # Produces report figures
├── model_ready/                       # Model-ready CSVs (3 windows x calls/puts)
├── figures/                           # Generated plots
├── checkpoints/                       # Saved model weights
├── .env.example                       # Template for DATABENTO_API_KEY
├── .gitignore
└── README.md
```

**Note on data:** only `model_ready/` is committed to this repository. The raw Brent/SOFR CSVs (`data/raw/` in the scripts' file paths) are not tracked in Git — they're either too large for a repo of this kind or straightforward to regenerate by re-running the extraction pipeline below. If you're cloning this repo fresh, you'll need to run steps 1–8 of the Pipeline section yourself before `model_ready/` can be rebuilt from scratch (the committed version is provided so the model can be trained/evaluated without needing your own Databento key).

## Data

- **Source:** Databento, CME Globex MDP 3.0 (GLBX.MDP3)
  - Brent EU-style options: root `BE` (underlying futures root `BZ`)
- **Features:** moneyness, time-to-maturity (years), realised volatility, SOFR rate
- **Label:** empirically-derived local hedge ratio (constructed from market data)
- **Rate series:** daily SOFR (FRED), used in place of a flat-rate placeholder
- **Label bounds:** minimum ΔF threshold of 0.50, labels capped at |2.0|

Row counts (Brent, all windows): 58,150 (prior shock) / 18,956 (calm) / 57,088 (test shock) — 134,194 total.

## Setup

```bash
git clone <repo-url>
cd final_project
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Databento API key:

```
DATABENTO_API_KEY=your_key_here
```

The key is loaded via `python-dotenv`; `.env` is git-ignored and should never be committed.

## Pipeline

Run scripts from `src/` in the following order:

1. `data_extraction.py` — pull raw data
2. `fix_futures_settle.py` — correct futures settlement price gaps
3. `add_sofr.py` — merge in real daily SOFR rate, recompute implied vol
4. `data_preparation.py` — clean, filter, and chronologically split
5. `construct_label.py` — build hedge-ratio labels
6. `check_label_distribution.py` — sanity-check labels
7. `multicollinearity_check.py` — verify feature independence (VIF)
8. `finalise_data.py` — write model-ready datasets to `model_ready/`
9. `train_model.py` — train calls/puts networks
10. `evaluate_model.py` — evaluate against Black-76
11. `check_generalisation.py` — compare validation vs. test MSHE
12. `generate_figures.py` — produce report figures

Diagnostic and ablation scripts (`inspect_arbitrage_violations.py`, `diagnose_regime_shift.py`, `diagnose_regime_no_rate.py`, `run_ablation.py`, `weight_ablation.py`, `train_modelwd0.py`, `evaluate_modelwd0.py`, `diagnose_dates.py`) are run standalone, as needed, rather than as part of the core numbered pipeline.

## Model

- Architecture: separate feedforward networks for calls and puts, sigmoid-bounded output
- Loss: Huber loss
- Regularisation: dropout (p=0.3), L2 weight decay (1e-4)

## Results (test window)

| Metric (MSHE) | Neural Network | Black-76 |
|---|---|---|
| Calls | 0.2107 | 0.0920 |
| Puts | 0.1430 | 0.0867 |

Black-76 outperforms the neural network on the held-out test window. Diagnostics indicate this reflects an information asymmetry rather than overfitting or horizon mismatch: Black-76 uses the market's current implied volatility, while the network relies only on backward-looking realised volatility. Adding implied volatility as a fifth feature worsened test performance (calls 0.29, puts 0.20 MSHE), consistent with overfitting.

## Notes

- `implied_vol` is derived by numerically inverting Black-76 against the observed settlement price; residual missingness (~2-5%) reflects contracts near the no-arbitrage boundary and is expected in options data.
- This repository supports the MSc dissertation submitted for IFTE0008 (Research Project, UCL MSc IFT programme).
