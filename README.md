# GENESIS-AI
Generative Ephemeris Error Synthesis & Intelligent Simulation

Predict GNSS (Global Navigation Satellite System) clock and ephemeris errors with a hybrid deterministic + generative AI pipeline.

**Mission:**  
Build a model that forecasts satellite clock/orbit error at 15-minute intervals for an unseen 8th day (15 min → 24 h horizons).

**Key novelties:**  
• Hybrid deterministic-generative model  
• Normality-aware loss (residuals close to Gaussian)  
• Horizon-adaptive ensemble (GRU + Transformer + GP head)  
• Uncertainty quantification and explainability dashboard  

**Environment:**  
Designed for Google Colab training and Streamlit deployment.

**Basic usage:**
```bash
pip install -e .
```

(Colab template in `notebooks/` guides full training and demo run.)
