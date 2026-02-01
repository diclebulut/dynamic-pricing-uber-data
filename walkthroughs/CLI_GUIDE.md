# Dynamic Pricing Analysis - CLI Guide

Complete command-line interface for running the dynamic pricing analysis pipeline with configurable options.

## Quick Start

### Basic Usage (All Defaults)
```bash
cd src
python pipeline.py
```

This runs the complete pipeline with:
- Full console output
- 100 DQN training episodes
- 20 test episodes
- No markdown/plot outputs (use flags to enable)

## Command-Line Options

### Output Control

#### `--verbose` (boolean, default: True)
Controls console output verbosity.
```bash
python pipeline.py --verbose True    #full
python pipeline.py --verbose False   #minimal
```

#### `--save-markdown` (flag)
Saves analysis results as a markdown report.
```bash
python pipeline.py --save-markdown
#creates "outputs/reports/pricing_analysis_YYYYMMDD_HHMMSS.md"
```

#### `--markdown-complexity` (choices: summary | standard | full)
Controls the detail level of markdown reports (default: full).
```bash
python pipeline.py --save-markdown --markdown-complexity summary
#quick executive summary

python pipeline.py --save-markdown --markdown-complexity standard
#include detailed metrics comparison

python pipeline.py --save-markdown --markdown-complexity full
#include all metrics, acceptance rates, and statistical summaries
```

#### `--save-plots` (flag)
Generates and saves visualization plots.
```bash
python pipeline.py --save-plots
#creates "outputs/visualizations/comparison_YYYYMMDD_HHMMSS.png"
```

### DQN Training Configuration

#### `--episodes` (integer, default: 100)
Number of DQN training episodes.
```bash
python pipeline.py --episodes 50   #quick
python pipeline.py --episodes 200  #thorough
```

#### `--batch-size` (integer, default: 32)
Experience replay batch size (experiences per training step).
```bash
python pipeline.py --batch-size 16   #small batch, more updates
python pipeline.py --batch-size 64   #large batch, stable updates
```

#### `--update-freq` (integer, default: 10)
Target network update frequency (episodes between updates).
```bash
python pipeline.py --update-freq 5   #more frequent
python pipeline.py --update-freq 20  #less frequent
```

#### `--test-episodes` (integer, default: 20)
Number of test episodes to evaluate the trained model.
```bash
python pipeline.py --test-episodes 10  #quick
python pipeline.py --test-episodes 50  #thorough
```

## Complete Examples

### 1. Full Analysis with Reports
```bash
python pipeline.py --save-markdown --markdown-complexity full --save-plots
```
Runs complete pipeline and saves detailed markdown report + visualizations.

### 2. Quick Analysis (Summary Only)
```bash
python pipeline.py --verbose False --episodes 50 --test-episodes 10
```
Fast run with minimal output for quick testing.

### 3. Detailed Training Analysis
```bash
python pipeline.py \
  --episodes 200 \
  --batch-size 32 \
  --update-freq 10 \
  --test-episodes 30 \
  --save-markdown \
  --markdown-complexity full \
  --save-plots
```
Comprehensive analysis with extensive training and full documentation.

### 4. Conservative Configuration
```bash
python pipeline.py \
  --episodes 150 \
  --batch-size 64 \
  --update-freq 15 \
  --test-episodes 25 \
  --save-markdown \
  --markdown-complexity standard
```
Stable training with balanced batch sizes and moderate output.

### 5. Aggressive Configuration
```bash
python pipeline.py \
  --episodes 300 \
  --batch-size 16 \
  --update-freq 5 \
  --test-episodes 40 \
  --save-markdown \
  --markdown-complexity full \
  --save-plots
```
Maximum training with frequent updates and comprehensive reporting.

## Output Structure

After running the pipeline, outputs are organized as:

```
outputs/
├── reports/
│   └── pricing_analysis_20240101_120000.md
└── visualizations/
    └── comparison_20240101_120000.png
```

### Markdown Report Contents

**Summary Level:**
- Revenue improvement percentage
- Acceptance rates (DQN vs Static)

**Standard Level:**
- All summary metrics
- Detailed comparison metrics
- Price standard deviations

**Full Level:**
- All standard metrics
- Acceptance rate comparison details
- Statistical summaries
- Performance metrics per ride type

### Visualization Contents

The PNG file includes 4 subplots:
1. **Acceptance Rate vs Price Multiplier** - Scatter plot showing pricing-acceptance relationship
2. **Reward Distribution** - Histogram of rewards achieved
3. **Predicted Price vs Actual Fare** - Comparison of pricing strategies
4. **Acceptance Rate Distribution** - Pie chart of accepted vs rejected rides

## Performance Tuning

### For Faster Execution
```bash
python pipeline.py \
  --verbose False \
  --episodes 50 \
  --test-episodes 10 \
  --batch-size 64
```
Expected runtime: ~2-3 minutes

### For Better Model Quality
```bash
python pipeline.py \
  --episodes 300 \
  --batch-size 16 \
  --update-freq 5 \
  --test-episodes 50
```
Expected runtime: ~15-20 minutes

### For Production Deployment
```bash
python pipeline.py \
  --episodes 500 \
  --batch-size 32 \
  --update-freq 10 \
  --test-episodes 100 \
  --save-markdown \
  --markdown-complexity full \
  --save-plots
```
Expected runtime: ~30-40 minutes

## Module Structure

### `modules/static.py`
- `calculate_acceptance_rate()` - Sigmoid-based acceptance model
- `train_static_model()` - Calibrate pricing on training data
- `apply_static_pricing()` - Apply static prices to test data
- `get_static_metrics()` - Extract performance metrics

### `modules/reinforcement.py`
- `DQN` - Neural network class for Q-learning
- `PricingEnvironment` - Simulated pricing environment
- `DQNAgent` - Reinforcement learning agent
- `train_dqn_agent()` - Training loop
- `test_dqn_agent()` - Evaluation on test set
- `get_dqn_metrics()` - Extract performance metrics

### `pipeline.py`
- `set_seeds()` - Reproducibility setup
- `prepare_feature_engineered_data()` - Feature processing
- `run_static_pricing_pipeline()` - Static pricing execution
- `run_dqn_pipeline()` - DQN training and testing
- `compare_strategies()` - Strategy comparison
- `generate_visualizations()` - Plot generation
- `save_markdown_report()` - Report generation
- `main()` - Pipeline orchestration

## Environment Variables

All parameters can be set via command-line arguments. No environment variables are required.

## Troubleshooting

### Out of Memory
Reduce batch size and test episodes:
```bash
python pipeline.py --batch-size 16 --test-episodes 10
```

### Slow Execution
Reduce training episodes and batch size:
```bash
python pipeline.py --episodes 50 --batch-size 64 --test-episodes 5
```

### No Output Files
Ensure you use the flags:
```bash
python pipeline.py --save-markdown --save-plots --markdown-complexity full
```

## Advanced Usage

You can also import and use the pipeline programmatically:

```python
from pipeline import main

results = main(
    verbose=True,
    save_markdown=True,
    markdown_complexity='full',
    save_plots=True,
    num_training_episodes=100,
    batch_size=32,
    update_frequency=10,
    test_episodes=20
)


static_metrics = results['static_metrics']
dqn_metrics = results['dqn_metrics']
comparison_df = results['comparison_df']
```

## Additional Scripts

### Run from main.py
```bash
python main.py
```
Uses default pipeline settings.

### Direct module usage
```python
from modules.preprocessing import prepare_columns
from modules.static import train_static_model, apply_static_pricing
from modules.reinforcement import PricingEnvironment, DQNAgent

data = prepare_columns()

price_per_mile = train_static_model(train_data)
static_results = apply_static_pricing(test_data, price_per_mile)
```

---

For more information, see the notebook at `src/new_data.ipynb` for interactive exploration of the analysis pipeline.
