# Mini-Challenge Model Source Code

This repository contains the implementation code for our model in the mini-challenge.

## Getting Started

### Prerequisites
1. **Preprocessing**:  
   Run the following scripts in order:
   - `00-preprocessing.py`
   - `01-Mini-challenge.py`  
   These scripts will preprocess all required datasets.

2. **Quantile Estimation**:  
   Execute `02-00-Quantile-of-conformityscore.py` to compute the empirical quantile of the nonconformity score.  

## Implementation Details

### Key Assumption
The quantile estimation step relies on an intuitive assumption:

If we train a model **after removing the 2 downstream stations** from the training set,  
the resulting nonconformity scores $ s_i \sim S_1 $ for these downstream stations  
and the scores $ s_i \sim S_2 $ computed using the full training set as well as the 2 stations in the minichallenge, are identically distributed:  
$$S_1 \overset{(d)}{=} S_2$$


### Methodology
1. Compute the empirical quantile of $ S_1 $ to guarantee coverage for the 2 downstream stations
2. Use this quantile to calibrate predictions for the mini-challenge stations
