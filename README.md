# 🏆 GENESIS-AI: Competition-Grade GNSS Error Prediction System

**Advanced AI/ML system for predicting GNSS satellite clock bias and ephemeris errors with normality-optimized distribution matching**

---

## 🎯 **Competition Problem Statement Alignment**

GENESIS-AI has been specifically designed and optimized for the **GNSS Error Prediction Competition** with the following key requirements:

### **Problem Focus**
- **Primary Target**: Predict differences between **broadcast** and **ICD-based modeled values** for both satellite clock biases and ephemeris parameters
- **Data Scope**: 7-day training data → Day-8 predictions at 15-minute intervals
- **Satellite Types**: Specialized handling for **GEO/GSO** and **MEO** satellites
- **Prediction Horizons**: 15min, 30min, 1hour, 2hour, 24hour forecasts

### **Competition Evaluation Criteria**
- **Primary Metric (70%)**: **Normal Distribution Closeness** - How closely prediction errors follow a normal distribution
- **Secondary Metric (30%)**: **Prediction Accuracy** - Traditional RMSE/MAE metrics
- **Success Criteria**: Models demonstrating robust performance across all horizons with optimal error distribution normality

---

## 🧠 **AI/ML Architecture Overview**

### **Dual Prediction System**
```
┌─────────────────────────────────────────────────────────┐
│                GENESIS-AI Architecture                   │
├─────────────────────────────────────────────────────────┤
│  📡 Data Input (Broadcast vs Modeled Values)           │
│    ├── Clock Bias Differences                          │
│    └── Ephemeris Differences (6D: dx,dy,dz,dvx,dvy,dvz)│
├─────────────────────────────────────────────────────────┤
│  🔄 Orbit-Specific Processing                          │
│    ├── GEO/GSO Models (24h period, high altitude)      │
│    └── MEO Models (12h period, GPS-like)               │
├─────────────────────────────────────────────────────────┤
│  🤖 AI/ML Models (Competition-Required)                │
│    ├── 🧠 Recurrent Neural Networks (GRU/LSTM)        │
│    ├── 🎭 Generative Adversarial Networks (GANs)      │
│    ├── 🔍 Transformers (Multi-scale Attention)        │
│    └── 📊 Gaussian Processes (Uncertainty)            │
├─────────────────────────────────────────────────────────┤
│  ⚖️ Normality-Aware Training (70% Weight)             │
│    ├── Kolmogorov-Smirnov Tests                        │
│    ├── Anderson-Darling Tests                          │
│    ├── Shapiro-Wilk Tests                              │
│    └── Histogram Entropy Optimization                  │
├─────────────────────────────────────────────────────────┤
│  📈 Multi-Horizon Predictions                          │
│    └── Specialized heads for each time horizon         │
└─────────────────────────────────────────────────────────┘
```

### **Core AI/ML Techniques**

#### **1. Physics-Informed Neural Networks**
- **Orbital Mechanics Integration**: Kepler's equations, relativistic effects, J2 perturbations
- **Satellite-Specific Physics**: Different physics for GEO/GSO vs MEO orbits
- **Real-time Corrections**: Dynamic physics-based error correction

#### **2. Advanced Transformer Architecture**
- **Multi-Scale Temporal Attention**: Captures patterns across different time scales
- **Constellation-Aware Processing**: Specialized attention for satellite constellations
- **Physics-Enhanced Positional Encoding**: Orbital mechanics-informed position encoding

#### **3. Generative Adversarial Networks**
- **Error Pattern Synthesis**: Generate realistic GNSS error patterns for data augmentation
- **Distribution Matching**: Train generator to produce normally-distributed errors
- **Adversarial Normality Training**: Discriminator optimizes for normal distribution

#### **4. Gaussian Process Uncertainty Quantification**
- **Bayesian Uncertainty**: Probabilistic error bounds for all predictions
- **Model Disagreement**: Ensemble uncertainty estimation
- **Confidence Intervals**: Horizon-specific uncertainty quantification

---

## 🚀 **Quick Start - Competition Mode**

### **Installation**
```bash
# Clone repository
git clone https://github.com/shashank6009/ISRO-SIH.git
cd genesis-ai

# Install dependencies
pip install -e .

# Install additional competition dependencies
pip install kaleido plotly scipy scikit-learn
```

### **Competition Data Format**
Your CSV file must contain these columns:
```csv
satellite_id,timestamp,orbit_class,broadcast_clock_bias,modeled_clock_bias,clock_error,
broadcast_x,broadcast_y,broadcast_z,broadcast_vx,broadcast_vy,broadcast_vz,
modeled_x,modeled_y,modeled_z,modeled_vx,modeled_vy,modeled_vz,
ephemeris_error_x,ephemeris_error_y,ephemeris_error_z,ephemeris_error_vx,ephemeris_error_vy,ephemeris_error_vz
```

### **Run Competition Dashboard**
```bash
# Start the competition system
npm run serve-competition

# Or manually:
python3 -m uvicorn genesis_ai.inference.service:app --host 0.0.0.0 --port 8000 &
python3 -m streamlit run src/genesis_ai/app/competition_dashboard.py --server.port=8502
```

### **Day-8 Prediction Pipeline**
```python
from genesis_ai.competition.day8_prediction_framework import run_day8_competition
from genesis_ai.db.competition_models import CompetitionDataLoader

# Load your 7+1 day competition data
loader = CompetitionDataLoader()
data_points = loader.prepare_competition_data(
    loader.load_competition_csv("your_competition_data.csv")
)

# Separate training (7 days) and test (day 8) data
training_data = [dp for dp in data_points if dp.is_training]
day8_data = [dp for dp in data_points if not dp.is_training]

# Run complete competition pipeline
results = run_day8_competition(training_data, day8_data)

print(f"Competition Score: {results['prediction_results']['evaluation_metrics'].final_score:.4f}")
print(f"Normality Score (70%): {results['prediction_results']['evaluation_metrics'].normality_score:.4f}")
print(f"Accuracy Score (30%): {results['prediction_results']['evaluation_metrics'].accuracy_score:.4f}")
```

---

## 📊 **Competition-Specific Features**

### **1. Normality-Optimized Training**
- **Primary Focus**: 70% weight on achieving normal error distribution
- **Multiple Statistical Tests**: KS, Anderson-Darling, Shapiro-Wilk, Jarque-Bera
- **Adaptive Loss Functions**: Dynamic weighting based on normality performance
- **Distribution Matching**: GAN-based normal distribution synthesis

### **2. Multi-Horizon Specialized Prediction**
- **15-minute**: Ultra-short term with high accuracy weight (30%)
- **30-minute**: Short term with balanced performance (25%)
- **1-hour**: Medium term with robust predictions (20%)
- **2-hour**: Extended term with uncertainty quantification (15%)
- **24-hour**: Long term with physics-informed corrections (10%)

### **3. Orbit-Class Specific Models**
- **GEO/GSO Satellites**: 
  - 24-hour orbital period optimization
  - Enhanced relativistic effect modeling
  - Solar pressure and eclipse handling
  - Higher clock stability assumptions
- **MEO Satellites**:
  - 12-hour orbital period optimization
  - J2 perturbation emphasis
  - Multi-path effect modeling
  - Dynamic ephemeris precision

### **4. Broadcast vs Modeled Value Processing**
- **Dedicated Data Schema**: Explicit handling of broadcast/modeled differences
- **Error Decomposition**: Separate clock bias and ephemeris error prediction
- **ICD Compliance**: Alignment with International Civil Aviation Organization standards
- **Real-time Validation**: Continuous broadcast/modeled value consistency checking

---

## 🏆 **Competition Advantages**

### **Technical Innovations**
1. **Normality-First Design**: Unlike traditional accuracy-focused models, GENESIS-AI prioritizes normal error distribution
2. **Physics-AI Fusion**: Combines orbital mechanics with deep learning for physically consistent predictions
3. **Multi-Scale Architecture**: Captures both short-term fluctuations and long-term orbital patterns
4. **Uncertainty-Aware**: Provides confidence intervals for all predictions

### **Performance Optimizations**
1. **Dual Predictor Architecture**: Specialized models for clock vs ephemeris errors
2. **Orbit-Specific Processing**: Tailored models for different satellite orbits
3. **Ensemble Learning**: Multiple diverse models with disagreement-based uncertainty
4. **Adaptive Training**: Dynamic loss weighting based on normality performance

### **Competition Compliance**
1. **Exact Problem Alignment**: Designed specifically for broadcast vs modeled value prediction
2. **Required AI/ML Methods**: Implements all competition-specified techniques (RNN, GAN, Transformer, GP)
3. **Day-8 Framework**: Specialized pipeline for 7-day training → day-8 prediction
4. **Evaluation Matching**: Uses competition-exact scoring (70% normality, 30% accuracy)

---

## 🔧 **System Components**

### **Core Modules**
- `src/genesis_ai/db/competition_models.py` - Competition data schema and loading
- `src/genesis_ai/models/competition_dual_predictor.py` - Dual clock/ephemeris predictors
- `src/genesis_ai/models/orbit_specific_models.py` - GEO/GSO vs MEO specialized models
- `src/genesis_ai/training/competition_loss.py` - Normality-prioritized loss functions
- `src/genesis_ai/evaluation/enhanced_competition_metrics.py` - Competition scoring system
- `src/genesis_ai/competition/day8_prediction_framework.py` - Day-8 prediction pipeline

### **AI/ML Models**
- `src/genesis_ai/physics/orbital_mechanics.py` - Physics-informed neural networks
- `src/genesis_ai/models/advanced_transformer.py` - Multi-scale temporal transformers
- `src/genesis_ai/models/gan_forecaster.py` - Generative adversarial networks
- `src/genesis_ai/models/master_competition_model.py` - Ensemble competition model

### **User Interface**
- `src/genesis_ai/app/competition_dashboard.py` - Professional competition dashboard
- Clean, dark-themed interface optimized for ISRO professional use
- Detailed performance analysis with comprehensive visualizations
- Real-time normality assessment and statistical test results

---

## 📈 **Expected Competition Performance**

### **Normality Scores (Primary - 70%)**
- **Target**: >0.85 composite normality score across all horizons
- **Methods**: 6 statistical tests (KS, AD, SW, JB, DP, Histogram)
- **Optimization**: GAN-enhanced distribution matching

### **Accuracy Scores (Secondary - 30%)**
- **15min RMSE**: <1e-6 for clock bias, <10m for ephemeris
- **24hour RMSE**: <5e-6 for clock bias, <100m for ephemeris
- **Cross-horizon Consistency**: Smooth degradation with horizon

### **Overall Competition Score**
- **Target Final Score**: >0.80 (combining 70% normality + 30% accuracy)
- **Ranking Goal**: Top 3 in competition leaderboard
- **Robustness**: Consistent performance across all satellite types and horizons

---

## 🔬 **Technical Specifications**

### **Model Architecture**
- **Input Dimensions**: 128-512 features (orbit-specific)
- **Hidden Layers**: 4-6 physics-informed layers
- **Attention Heads**: 8-16 multi-head attention
- **Output Dimensions**: 1 (clock) + 6 (ephemeris) per horizon

### **Training Configuration**
- **Loss Function**: 70% normality + 30% accuracy weighted
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 1e-4 with cosine annealing
- **Batch Size**: 32 (orbit-specific batching)
- **Early Stopping**: 15 epochs patience on validation normality

### **Performance Requirements**
- **Training Time**: <2 hours on GPU for 7-day dataset
- **Inference Speed**: <100ms for all horizons
- **Memory Usage**: <8GB GPU memory
- **Scalability**: Handles 1000+ satellites simultaneously

---

## 📚 **Documentation**

- `docs/architecture.md` - System architecture and design decisions
- `docs/model_card.md` - Model specifications and performance metrics
- `COMPETITION_README.md` - Detailed competition system documentation
- `UI_TRANSFORMATION_SUMMARY.md` - User interface design and features

---

## 🤝 **Contributing**

This system is optimized for the GNSS Error Prediction Competition. For contributions:

1. **Competition Focus**: Ensure changes improve normality scoring or accuracy
2. **Physics Compliance**: Maintain orbital mechanics consistency
3. **Performance**: Optimize for Day-8 prediction accuracy
4. **Documentation**: Update competition-specific documentation

---

## 📄 **License**

MIT License - See `LICENSE` file for details.

---

## 🏅 **Competition Team**

**GENESIS-AI Development Team**
- Advanced AI/ML architecture for GNSS error prediction
- Physics-informed neural network design
- Competition-optimized normality scoring
- Professional ISRO-grade user interface

**Built for Excellence in GNSS Error Prediction Competition** 🚀