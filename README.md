<div align="center">

# Hypix AI Trading Assistant

**Hybrid AI Trading System with Unity ML-Agents and Streamlit Interface**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Unity ML-Agents](https://img.shields.io/badge/Unity-ML--Agents-black.svg)](https://unity.com/products/machine-learning-agents)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org)
[![License](https://img.shields.io/badge/License-Proprietary-critical.svg)](#)

---

</div>

## Overview

Hypix is a hybrid AI trading system combining Unity ML-Agents reinforcement learning with traditional deep learning models. The platform features a Streamlit web interface for market analysis while leveraging trained RL agents for decision-making and strategy optimization.

The system integrates GGUF-format language models, custom H5 neural networks, and ensemble ML architectures to generate trading signals with 50+ technical indicators.

## Key Features

### Unity ML-Agents Integration
- **GGUF Model**: Primary inference model (`Hypix.gguf`) for efficient decision-making
- **Reinforcement Learning Agents**: Trained H5 models for market environment interaction
  - `Hyphix_model1.h5` - Primary trading agent
  - `Hyphix_model2.h5` - Secondary strategy agent
- **JavaScript Engine**: Core and evaluator engines for runtime execution
- **Training Pipeline**: JSON-based training data for continuous model improvement

### Multi-Model AI Architecture
- **Transformer Network** with multi-head attention mechanism
- **Bidirectional LSTM** with attention layers for sequence modeling
- **CNN-LSTM Hybrid** combining spatial and temporal feature extraction
- **XGBoost** gradient boosting for pattern recognition
- **Random Forest** ensemble classifier
- **Gradient Boosting** classifier
- Weighted voting ensemble for consensus predictions

### Technical Analysis Suite
50+ technical indicators including:
- Moving averages (SMA, EMA)
- Momentum indicators (RSI, MACD, Stochastic, Williams %R)
- Volatility measures (Bollinger Bands, ATR, ADX)
- Volume indicators (OBV, MFI, VWAP)
- Custom pattern recognition algorithms

### Real-Time Market Data
- Integration with yfinance for live market data
- Support for multiple data providers (Polygon.io, Finnhub, IEX Cloud)
- Automatic failover and caching mechanisms
- Multiple timeframe support (intraday, daily, weekly)

### Interactive Visualizations
- Candlestick charts with technical overlays
- Multi-panel indicator displays
- Volume analysis charts
- Prediction confidence gauges
- Comparative stock analysis

### Trading Intelligence
- Three-class predictions (DOWN/NEUTRAL/UP)
- Confidence scoring for each prediction
- Entry/exit point recommendations
- ATR-based stop-loss calculations
- Risk-reward ratio optimization
- Multi-stock comparison and ranking

## Installation

### Prerequisites

```bash
Python 3.8 or higher
Unity ML-Agents Toolkit
pip package manager
```

### Dependencies

Install required packages:

```bash
pip install streamlit pandas numpy tensorflow scikit-learn xgboost plotly yfinance ta-lib python-dotenv requests mlagents
```

For Unity ML-Agents development:
```bash
pip install mlagents==0.30.0
pip install torch torchvision
```

**Note:** TA-Lib requires system-level installation:

**macOS:**
```bash
brew install ta-lib
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ta-lib
```

**Windows:**
Download pre-built wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hypix.git
cd hypix
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the application:
```bash
streamlit run hypix_streamlit.py
```

4. Open your browser to `http://localhost:8501`

## Usage

### Web Interface

The Streamlit dashboard provides three main sections:

**1. Stock Analysis**
- Select trading mode (Day Trading, Swing Trading, Long-term)
- Choose from 20 pre-categorized stocks by volatility
- View AI predictions with confidence scores
- Analyze technical indicators and patterns
- Get specific entry/exit recommendations

**2. Multi-Stock Comparison**
- Compare up to 5 stocks simultaneously
- View ranked recommendations
- Analyze relative technical strength
- Identify best trading opportunities

**3. Trading Education**
- Interactive Q&A interface
- Beginner and intermediate cheat sheets
- Technical indicator explanations
- Pattern recognition guides

### Configuration

Stock categories are organized by volatility tiers:

**High Volatility:** TSLA, NVDA, AMD, COIN, MSTR, RIVN, PLTR
**Medium Volatility:** AAPL, MSFT, GOOGL, META, AMZN, NFLX, DIS
**Low Volatility:** JNJ, PG, KO, WMT, PEP, VZ

Modify categories in the `HypixConfig` class within the source code.

### API Configuration (Optional)

(Hypix accepts yfinance with User-Agent.) While yfinance works without API keys, you can add additional data sources:

Create a `.env` file in the project root: 

```env
POLYGON_API_KEY=your_polygon_key
FINNHUB_API_KEY=your_finnhub_key
IEX_API_KEY=your_iex_key
```

Free tier limits:
- **Polygon.io:** 5 calls/minute
- **Finnhub:** 60 calls/minute
- **IEX Cloud:** 50,000 credits/month

## System Architecture

### Project Structure

```
Hypix/
├── Primary Model/
│   └── Hypix.gguf                    # GGUF inference model
├── Resources/
│   ├── core_engine.main.js           # Core execution engine
│   └── evaluator_engine.main.js      # Model evaluation engine
├── Trained h5 Models/
│   ├── Hypix_model1.h5             # Primary RL agent
│   └── Hypix_model2.h5             # Secondary RL agent
├── Training Data/
│   └── training.json                 # Training dataset
├── hypix_streamlit.py                # Streamlit web interface
└── hypix_env                         # Unity ML environment
```

### Data Pipeline
```
Market Data → Preprocessing → Feature Engineering → RL Agent → Model Inference → Analysis
     ↓              ↓                  ↓              ↓             ↓             ↓
  yfinance    Normalization    50+ Indicators   Unity ML     GGUF/H5 Models  Streamlit UI
```

### Model Training Pipeline
```
Historical Data → Unity Environment → RL Training → Model Checkpointing
                         ↓                 ↓
                  Reward Function    PPO/SAC Algorithm
                         ↓                 ↓
                  training.json      H5 Model Export
```

### Prediction Flow
```
Real-time Data → Indicator Calculation → RL Agent Decision → GGUF Inference → 
Ensemble Voting → Signal Generation → Risk Management → Trading Recommendations
```

## Technical Specifications

**Model Architectures:**
- LSTM Units: [256, 128, 64]
- Attention Heads: 8
- Transformer Blocks: 4
- Dropout Rate: 0.3
- Batch Size: 64
- Training Epochs: 100 (with early stopping)

**Data Processing:**
- Validation Split: 20%
- Feature Scaling: StandardScaler & RobustScaler
- Time Windows: 60 bars for short-term, 200 bars for long-term
- Cache Duration: 5 minutes

**Performance Metrics:**
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix analysis
- Multi-class ROC curves
- Prediction confidence intervals

## Example Workflow

### Day Trading Analysis

```python
# The system handles this through the UI, but programmatically:
from hypix_streamlit import RealTimeDataProvider, HypixAI, HypixAssistant

# Initialize components
data_provider = RealTimeDataProvider()
ai_model = HypixAI(HypixConfig())
assistant = HypixAssistant(data_provider, ai_model)

# Fetch and analyze
df = data_provider.get_stock_data('TSLA', 'day_trading')
ai_model.train(df)
analysis = assistant.analyze_stock('TSLA', 'day_trading')

print(f"Prediction: {analysis['prediction']}")
print(f"Confidence: {analysis['confidence']:.1f}%")
print(f"Entry: ${analysis['entry_price']:.2f}")
print(f"Stop Loss: ${analysis['stop_loss']:.2f}")
print(f"Target: ${analysis['target']:.2f}")
```

## Performance Considerations

**Computational Requirements:**
- RAM: Minimum 8GB, recommended 16GB
- GPU: Optional but recommended for model training (CUDA-compatible)
- CPU: Multi-core processor for parallel model execution

**Data Caching:**
The system implements intelligent caching with 5-minute expiration to balance data freshness with API rate limits.

**Model Persistence:**
Trained models are automatically saved and loaded to avoid retraining on every analysis.

## Limitations and Disclaimers

### Technical Limitations
- Historical data availability depends on third-party providers
- Model predictions are probabilistic, not deterministic
- Performance varies based on market conditions and volatility
- Computational requirements may limit real-time performance on low-end hardware

### Investment Disclaimer

> **CRITICAL NOTICE:** This software is provided for educational and analytical purposes only. It does not constitute financial advice, investment recommendations, or a solicitation to buy or sell securities.

**Trading involves substantial risk of loss. Key considerations:**
- Past performance does not guarantee future results
- AI predictions are based on historical patterns and may not reflect future market behavior
- Users are solely responsible for all trading decisions
- Consult qualified financial professionals before making investment decisions
- The developers assume no liability for financial losses

**Regulatory Compliance:** Users must ensure compliance with applicable securities regulations in their jurisdiction, including but not limited to SEC, FINRA, and state-specific requirements.

## Development

### Code Structure

```
Hypix Project
├── Primary Model
│   └── Hypix.gguf                    # Quantized inference model
├── Resources
│   ├── core_engine.main.js           # JavaScript execution runtime
│   └── evaluator_engine.main.js      # Model evaluation and scoring
├── Trained h5 Models
│   ├── Hypix_model1.h5             # Reinforcement learning agent 1
│   └── Hypix_model2.h5             # Reinforcement learning agent 2
├── Training Data
│   └── training.json                 # Historical market training data
└── hypix_streamlit.py                # Main application
    ├── Configuration (HypixConfig)
    ├── Data Acquisition (RealTimeDataProvider)
    ├── Technical Analysis (TechnicalAnalyzer)
    ├── AI Models (HypixAI)
    │   ├── Transformer
    │   ├── LSTM
    │   ├── CNN-LSTM
    │   ├── XGBoost
    │   ├── Random Forest
    │   └── Gradient Boosting
    ├── RL Agent Integration (Unity ML-Agents)
    ├── Trading Logic (HypixAssistant)
    ├── UI Components (HypixWebApp)
    └── Main Execution
```

### Unity ML-Agents Integration

**GGUF Model Format:**
The primary model uses GGUF (GPT-Generated Unified Format) for efficient inference with reduced memory footprint.

**H5 Model Loading:**
```python
from tensorflow import keras

# Load trained RL agents
model1 = keras.models.load_model('Trained h5 Models/Hypix_model1.h5')
model2 = keras.models.load_model('Trained h5 Models/Hypix_model2.h5')

# Use for prediction
prediction = model1.predict(observation)
```

**Training Data Format:**
```json
{
  "episodes": [
    {
      "observations": [...],
      "actions": [...],
      "rewards": [...],
      "timestamp": "2026-01-30T16:34:38Z"
    }
  ]
}
```

**JavaScript Engine Integration:**
The core and evaluator engines provide runtime execution for model inference and performance evaluation.

### Extending the System

**Adding New Indicators:**
Modify the `TechnicalAnalyzer.calculate_indicators()` method:

```python
def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    # Add your custom indicator
    df['CUSTOM_IND'] = your_calculation(df)
    return df
```

**Adding New Stocks:**
Update `HypixConfig.STOCKS` dictionary:

```python
STOCKS = {
    VOLATILITY_HIGH: ['TSLA', 'NVDA', 'YOUR_STOCK'],
    # ...
}
```

**Custom Models:**
Implement new models in the `HypixAI` class following the existing pattern.

**Training New RL Agents:**
```bash
# Using Unity ML-Agents
mlagents-learn config/hypix_config.yaml --run-id=hypix_training

# Export to H5 format
python export_model.py --checkpoint-path results/hypix_training --output Hypix_model3.h5
```

## Model Files

### GGUF Model (Hypix.gguf)
- **Format**: GGUF (GPT-Generated Unified Format)
- **Purpose**: Primary inference model with quantized weights
- **Size**: Optimized for deployment (reduced from original FP32)
- **Usage**: Real-time market prediction and decision generation

### H5 Models
**Hypix_model1.h5**
- Primary reinforcement learning agent
- Trained on historical market data from training.json
- Action space: BUY, SELL, HOLD signals
- State space: Technical indicators + price history

**Hypix_model2.h5**
- Secondary RL agent for strategy diversification
- Alternative action policy for ensemble decisions
- Risk-adjusted reward function optimization

### Training Data (training.json)
Structured dataset containing:
- Historical price observations
- Action sequences
- Reward trajectories
- Episode metadata
- Market conditions and context

**Format Specification:**
```json
{
  "metadata": {
    "version": "1.0",
    "episodes": 1000,
    "timesteps": 50000
  },
  "data": [...]
}
```

## Troubleshooting

**Common Issues:**

1. **Unity ML-Agents connection errors**
   - Verify mlagents package installation
   - Check Unity environment compatibility
   - Ensure proper port configuration

2. **GGUF model loading fails**
   - Verify model file integrity
   - Check GGUF library version compatibility
   - Ensure sufficient memory allocation

3. **H5 model compatibility issues**
   - Models require TensorFlow 2.x
   - Check Keras version compatibility
   - Verify custom layer definitions if present

4. **TA-Lib installation fails**
   - Install system-level TA-Lib before pip package
   - Use pre-built wheels on Windows

5. **Streamlit connection errors**
   - Check firewall settings
   - Verify port 8501 availability
   - Use `--server.port` flag to change port

6. **Slow model training**
   - Reduce epoch count in HypixConfig
   - Use GPU acceleration
   - Reduce dataset size for testing

7. **API rate limit errors**
   - Enable data caching (default: ON)
   - Reduce analysis frequency
   - Upgrade to paid API tiers

8. **JavaScript engine errors**
   - Verify Node.js installation
   - Check engine file permissions
   - Review console logs for runtime errors

## Support and Contact

**Technical Issues:**
Open an issue on the GitHub repository

**Response Time:**
Community support typically within 48-72 hours


## License

© 2025 Red Rook, LLC. All Rights Reserved.

---

<div align="center">

**Version 1.0.0** • Built with Python, TensorFlow, and Streamlit

*Last Updated: January 30, 2026*

</div>
