# Ensembles of offline changepoint detection methods
![python](https://img.shields.io/badge/python-3.7%20|%203.8-blue)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

__This repository is a fork from the [original code](https://github.com/YKatser/CPDE) coming along the paper cited in the [Citation](#citation) section. It aims at cleaning the code, fixing some issues that were pointed out in the original repository and providing new experiments.__

Two Jupyter Notebooks are provided to reproduce the results that are presented in the [Leaderboard for TEP benchmark](#leaderboard-for-tep-benchmark) and the [Leaderboard for SKAB](#leaderboard-for-skab). They can be directly launched in Google Colab from here:
- <a href="https://colab.research.google.com/github/theovincent/CPDE/blob/make_ipynb_working/TEP_experiments.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> for TEP dataset.
- <a href="https://colab.research.google.com/github/theovincent/CPDE/blob/make_ipynb_working/SKAB_experiments.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> for SKAB dataset

The *search methods* (CPD algorithms) developped in this repository are extensions of the [*ruptures*](https://github.com/deepcharles/ruptures) python library. The *cost functions* used for ensemble models are:
- ar(1)
- mahalanobis
- l1
- l2
- rbf

For each dataset, the _ensemble bound_ have been computed. It is defined as the average score on the dataset of the best single cost predictor. This means that for each signal, we take the best score among the single cost predictors. In the original paper, *Dynp* is called *Opt*.

A [report](./report.pdf) has been made for readers that are willing to go further.

## Leaderboard for TEP benchmark
Sorted by NAB (standard); for all metrics bigger is better.
The current leaderboard is obtained with the window size for the NAB detection algorithm equal to 10% of the dataset length.
In parenthesis, the aggregation function and the scaling function is given for the ensemble models.
| Algorithm                                 | NAB (standard) | NAB (lowFP) | NAB (LowFN) |
|---                                        |---    |---    |---    |
Perfect detector                            | 100   | 100   | 100 
Dynp and Binseg - Ensemble bound*           | 42.28 | 41.75 | 42.47
DynpEnsemble and BinSegEnsemble* (Min+Rank) | 41.79 | 40.74 | 42.15
Win - Ensemble bound                        | 40.13 | 39.62 | 41.04
Dynp and Binseg* (Mahalanobis)              | 36.88 | 35.82 | 37.29
WinEnsemble (Max+Znorm)                     | 36.47 | 36.07 | 37.01
Win (rbf)                                   | 28.96 | 28.05 | 32.00
Null detector                               | 0     | 0     | 0

\*Dynp and Binseg have the same results on the TEP dataset because all the signals have only one changepoint which means that the *search methods* are the same.


## Leaderboard for SKAB
*Sorted by NAB (standard); for all metrics bigger is better.*  
*The current leaderboard is obtained with the window size for the NAB detection algorithm equal to 30 sec.*  
| Algorithm               | NAB (standard) | NAB (lowFP) | NAB (LowFN) |
|---                      |---    |---    |---    |
Perfect detector          | 100   | 100   | 100 
Binseg - Ensemble bound*  | 30.35 | 27.91 | 31.77
Win - Ensemble bound      | 29.37 | 27.79 | 30.09
Dynp - Ensemble bound*    | 28.34 | 26.00 | 29.66
BinSeg* (Mahalanobis)     | 24.10 | 21.69 | 25.04
Dynp* (Mahalanobis)       | 22.37 | 19.90 | 23.37
DynpEnsemble* (Max+Raw)   | 21.64 | 19.21 | 22.63
BinSegEnsemble* (Max+Raw) | 20.86 | 18.20 | 22.37
WinEnsemble (Min+MinAbs)  | 19.61 | 17.70 | 20.25
Win (l1)                  | 18.40 | 16.22 | 19.19
Null detector             | 0     | 0     | 0

\*For the sake of reducing time complexity, the admissible changepoints were computed every 5 timesteps (`jump=5`).

### Citation
To cite this work in your publications (APA format):
```
Katser, I., Kozitsin, V., Lobachev, V., & Maksimov, I. (2021). Unsupervised Offline Changepoint Detection Ensembles. Applied Sciences, 11(9), 4280.
```
Or in BibTeX format:
```
@article{katser2021unsupervised,
  title={Unsupervised Offline Changepoint Detection Ensembles},
  author={Katser, Iurii and Kozitsin, Viacheslav and Lobachev, Victor and Maksimov, Ivan},
  journal={Applied Sciences},
  volume={11},
  number={9},
  pages={4280},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

### Used materials and 3rd party code
The experiment is based on the [*ruptures*](http://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/index.html) library (Copyright (c) 2017, ENS Paris-Saclay, CNRS. All rights reserved.) and the paper "Selective review of offline change point detection methods. Signal Processing" by C. Truong, L. Oudre, N. Vayatis. [[paper]](https://www.sciencedirect.com/science/article/pii/S0165168419303494?via%3Dihub)

The *ruptures* python package is distributed under the following conditions:
```
BSD 2-Clause License
Copyright (c) 2017, ENS Paris-Saclay, CNRS. All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
