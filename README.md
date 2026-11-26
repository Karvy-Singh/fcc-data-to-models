# FCC Data Sampling and Modelling

This project evaluates whether 1,000 diversity-preserved observations can
represent a 49,935-row Fluid Catalytic Cracking (FCC) dataset for multi-output
regression.

## Model Structure

- Inputs: Excel columns `A:G` (7 operating variables)
- Outputs: `T:AA`, `AC:AI`, and `AO` (16 targets)
- Final selected rows: 1,000
- Untouched test rows: 9,987

Downstream mass flows and derived process quantities are intentionally excluded
from the inputs. Including them produces target reconstruction and unrealistically
high scores rather than genuine forward prediction.

## Diverse Row Selection

The test set is reserved before sampling. The sampler then operates only on the
remaining 39,948-row training pool:

1. Select and protect the training-pool minimum and maximum of every target.
2. Enumerate the 21 input pairs and greedily select rows until every observed
   input-level pair is covered.
3. Calculate proportional quotas for every individual input level using the
   largest-remainder rule.
4. Fill the sample using a seeded candidate search that prioritises deficient
   levels and penalises quota overshoot.
5. Apply quota-improving row swaps without removing protected extrema or pair
   coverage anchors.

The final sample retains:

- 34/34 individual input levels
- 491/491 pairwise input-level combinations
- 16/16 output ranges
- Maximum marginal input-proportion deviation of approximately 0.069 percentage
  points

## Subset Comparison

Preliminary shared-holdout results showed that random sampling already covers
all input levels and pairs, but diversity selection improves several nonlinear
models.

| Model | Random 1,000 R² | Diverse 1,000 R² | Full training R² |
|---|---:|---:|---:|
| Linear regression | 0.8614 | 0.8610 | 0.8621 |
| Quadratic regression | 0.9712 | 0.9695 | 0.9721 |
| K-nearest neighbours | 0.8670 | 0.8635 | 0.9680 |
| Random forest | 0.9803 | 0.9819 | 0.9995 |
| Extra trees | 0.9794 | 0.9825 | 0.9994 |
| Neural network | 0.9958 | 0.9967 | 0.9998 |

The full dataset retains the lowest prediction error. The 1,000-row subset is a
compact approximation, not a lossless replacement.

## Model Results

Models were tuned by five-fold cross-validation on the selected 1,000 rows and
evaluated on the untouched test set.

| Model | CV R² | Test R² | Distant-test R² |
|---|---:|---:|---:|
| Linear regression | 0.8537 | 0.8582 | 0.8702 |
| Polynomial regression | 0.9919 | 0.9940 | 0.9939 |
| Ridge regression | 0.8537 | 0.8582 | 0.8702 |
| Decision tree | 0.9577 | 0.9544 | 0.9393 |
| Random forest | 0.9776 | 0.9809 | 0.9766 |
| Extra trees | 0.9801 | 0.9838 | 0.9784 |
| Gradient boosting | **0.9962** | **0.9977** | **0.9965** |
| Neural network | 0.9941 | 0.9952 | 0.9942 |

Gradient boosting is most accurate when every input level is represented.

## Reliability Tests

The complete split, sampling, and training process was repeated for 10 seeds:

| Model | Mean test R² | Standard deviation | Worst run R² |
|---|---:|---:|---:|
| Polynomial regression | 0.99381 | 0.00017 | 0.99350 |
| Gradient boosting | **0.99741** | **0.00037** | **0.99669** |
| Neural network | 0.99603 | 0.00071 | 0.99518 |

When complete input levels were removed from training:

| Model | Unseen-level interpolation R² | Extrapolation R² |
|---|---:|---:|
| Polynomial regression | 0.1788 | -4.8932 |
| Gradient boosting | -0.8463 | -1.8519 |
| Neural network | **0.9263** | **-0.2958** |

The neural network is substantially safer for unseen in-range levels. None of
the tested models is reliable outside the observed input range.

Additional checks found:

- No negative predictions
- No yield or conversion predictions outside 0--100%
- Mean predicted yield sum: 100.002%
- Mean absolute yield-sum error: 0.146 percentage points
- Nominal 90% conformal interval coverage: 89.87%

## Final Model

The **neural network is the selected primary model** because it is second-best
under represented operating conditions and clearly best for unseen-level
interpolation. Its selected configuration is:

- Hidden layers: `(64, 64)`
- Activation: ReLU
- L2 regularisation: `alpha=0.01`
- Input and target scaling
- Early stopping

Gradient boosting remains a secondary reference and disagreement check. Large
disagreement between the two models should be treated as an uncertainty warning.

## Run

Create an environment and install the dependencies:

```bash
python -m venv .venv
.venv/bin/pip install numpy scikit-learn
```

Regenerate the final 1,000 rows and benchmark all models:

```bash
.venv/bin/python -B model_benchmark.py data/FCC_NN_Data_Set2.xlsx
```

Run the extended reliability tests:

```bash
.venv/bin/python -B reliability_tests.py data/FCC_NN_Data_Set2.xlsx
```
## Scope

The final neural network is supported for interpolation within the represented
FCC operating domain. Predictions outside the recorded minimum and maximum
input values must be rejected or explicitly flagged as extrapolation.
