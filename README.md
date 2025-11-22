# GENESIS-AI

### GNSS Error Prediction System for Broadcast–Model Discrepancy Forecasting

### Developed by **N. S. Shashank**

GENESIS-AI is a competition-grade, research-level system engineered to model and predict GNSS satellite clock bias and ephemeris errors with a strong emphasis on **normality-optimized error distribution**, as required by the GNSS Error Prediction Competition.
It integrates advanced AI/ML methods with physics-informed orbital mechanics to deliver a complete, production-ready prediction framework that performs at scale and maintains statistical rigor across all forecasting horizons.

This repository contains the full training, inference, evaluation, and visualization pipeline used for the ISRO competition prototype.

---

## 1. Competition Problem Alignment

GENESIS-AI is purpose-built to address the official GNSS competition objective:

**Primary Task**
Predict the difference between broadcast navigation data and ICD-modeled values for:

* Satellite clock bias error
* Ephemeris error across six state parameters (x, y, z, vx, vy, vz)

**Data Format**
The system is structured around the competition’s 7-day training + Day-8 prediction workflow with 15-minute interval forecasts.

**Satellite Classes**

* GEO/GSO
* MEO

Each class has independent modeling pipelines due to fundamentally different orbital mechanics and noise signatures.

**Evaluation Criteria**

* 70% weight: Closeness of error distribution to a true normal distribution
* 30% weight: Prediction accuracy (RMSE/MAE)

GENESIS-AI directly optimizes for this scoring system rather than using generic forecasting metrics.

---

## 2. System Architecture Overview

GENESIS-AI uses a dual-stream prediction architecture:
one for **clock bias error** and one for **ephemeris state error**, each backed by physics-aware modules and multi-scale temporal models.

```
Data Input (Broadcast vs ICD-modeled)
     ├── Clock bias differences
     └── Ephemeris differences (6-D state error)

Orbit-Specific Processing
     ├── GEO/GSO specialized models
     └── MEO specialized models

AI/ML Core Modules
     ├── Recurrent models (GRU/LSTM)
     ├── Transformer-based sequence models
     ├── Generative Adversarial Networks for distribution alignment
     └── Gaussian Process regression for uncertainty modeling

Normality-Aware Optimization Layer
     ├── KS, AD, SW, JB tests
     └── Distribution-matching objective functions

Prediction Heads
     └── Multi-horizon outputs: 15m, 30m, 1h, 2h, 24h
```

The system is intentionally modular to allow fast experimentation during the competition and clear interpretability for scientific reporting.

---

## 3. Core Technical Features

### 3.1 Physics-Informed Modeling

GENESIS-AI integrates orbital dynamics such as:

* Keplerian motion constraints
* J2 perturbations
* Relativistic corrections
* Solar radiation pressure models
* Orbit-specific periodicity (12-hour MEO, 24-hour GEO/GSO)

The model does not blindly learn; it incorporates physical priors to reduce drift over long prediction horizons.

### 3.2 Transformer-Driven Temporal Modeling

A custom multi-scale temporal attention module captures signal behavior ranging from high-frequency fluctuations to long-period orbital cycles.

### 3.3 GAN-Based Error Distribution Shaping

A generative adversarial component is embedded into the training loop to force prediction residuals toward a normal distribution.
This directly improves the 70% normality component of the competition scoring system.

### 3.4 Bayesian and GP Uncertainty Estimation

Gaussian Processes and ensemble disagreement layers provide interval-bounded confidence estimates at each forecast horizon.

---

## 4. Competition Workflow

### 4.1 Installation

```
git clone https://github.com/shashank6009/ISRO-SIH.git
cd genesis-ai
pip install -e .
pip install kaleido plotly scipy scikit-learn
```

### 4.2 Required CSV Schema

The prediction system requires the following fields:

```
satellite_id, timestamp, orbit_class,
broadcast_clock_bias, modeled_clock_bias, clock_error,
broadcast_x, broadcast_y, broadcast_z, broadcast_vx, broadcast_vy, broadcast_vz,
modeled_x, modeled_y, modeled_z, modeled_vx, modeled_vy, modeled_vz,
ephemeris_error_x, ephemeris_error_y, ephemeris_error_z,
ephemeris_error_vx, ephemeris_error_vy, ephemeris_error_vz
```

### 4.3 Execute Day-8 Prediction Pipeline

```
from genesis_ai.competition.day8_prediction_framework import run_day8_competition
from genesis_ai.db.competition_models import CompetitionDataLoader

loader = CompetitionDataLoader()
data_points = loader.prepare_competition_data(
    loader.load_competition_csv("your_competition_data.csv")
)

training_data = [dp for dp in data_points if dp.is_training]
day8_data     = [dp for dp in data_points if not dp.is_training]

results = run_day8_competition(training_data, day8_data)
```

Outputs include:

* Final competition score
* Normality-score
* Accuracy-score
* Full residual statistics and multi-horizon error curves

---

## 5. Competition-Driven Feature Advantages

1. **Normality-First Training Strategy**
   The system is explicitly engineered to shape prediction residuals into a normal distribution using multi-test optimization and GAN-driven distribution matching.

2. **Orbit-Specific Model Branching**
   GEO/GSO and MEO satellites are processed through separate pipelines to avoid mixing fundamentally different signal behaviors.

3. **Dual-Predictor Architecture**
   Independent predictors for clock bias and ephemeris errors optimize for the dynamics of each variable type.

4. **Ensemble-Based Stability**
   Multiple model families (RNN, Transformer, GAN-corrector, GP) improve robustness, especially for long-range forecasts.

5. **High-Fidelity Dashboard**
   A professional, ISRO-facing dashboard built using Streamlit and Plotly provides:

   * Normality diagnostics
   * Error histograms
   * Multi-horizon forecast comparison
   * Orbit-specific performance breakdown

---

## 6. Technical Specifications

* Input Vector Size: 128–512 features (dependent on orbit class)
* Hidden Layers: 4–6 physics-augmented layers
* Multi-Head Attention: 8–16 heads
* Horizon Outputs: Clock (1) + Ephemeris (6) for each forecast horizon
* Optimizer: AdamW
* LR Schedule: Cosine annealing
* Batch Size: 32
* Early Stopping: Based on validation normality score

**Performance Targets**

* Normality score: >0.85
* 15-minute RMSE:

  * Clock: <1e-6
  * Ephemeris: <10 m
* 24-hour RMSE:

  * Clock: <5e-6
  * Ephemeris: <100 m

---

## 7. Codebase Overview

```
src/genesis_ai/
    db/competition_models.py              # Data schema & loaders
    competition/day8_prediction_framework.py
    models/competition_dual_predictor.py  # Dual model architecture
    models/orbit_specific_models.py       # GEO/MEO processing
    models/advanced_transformer.py        # Transformer implementation
    models/gan_forecaster.py              # GAN error-distribution modules
    physics/orbital_mechanics.py          # Physics-informed layers
    training/competition_loss.py          # Normality-weighted loss
    evaluation/enhanced_competition_metrics.py
    app/competition_dashboard.py          # Visualization dashboard
```

Each component is optimized specifically for the GNSS competition and not generic ML use.

---

## 8. Documentation

* `docs/architecture.md` – architectural breakdown
* `docs/model_card.md` – model specification and performance
* `COMPETITION_README.md` – competition-aligned notes
* `UI_TRANSFORMATION_SUMMARY.md` – dashboard and design decisions

---

## 9. Contribution Guidelines

This repository is tightly tuned for GNSS error forecasting under competition constraints.
Any contributions should:

* Improve normality scoring
* Improve prediction accuracy
* Maintain physics consistency
* Maintain competition-compliant architecture

---

## 10. License

This project is released under the MIT License.
Refer to the LICENSE file for details.

---

## 11. Author

**N. S. Shashank**
Creator and Lead Developer, GENESIS-AI
Developed for the Indian Space Research Organization (ISRO) problem statement as part of the Smart India Hackathon.
