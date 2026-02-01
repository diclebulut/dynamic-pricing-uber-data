# Quick Reference - Dynamic Pricing CLI Commands

## Most Common Commands

### 1. Run Everything with Full Report
```bash
python src/pipeline.py --save-markdown --markdown-complexity full --save-plots
```

### 2. Quick Test Run
```bash
python src/pipeline.py --episodes 50 --test-episodes 10 --verbose False
```

### 3. Standard Analysis
```bash
python src/pipeline.py --save-markdown --save-plots
```

### 4. Training Only (no output files)
```bash
python src/pipeline.py --episodes 200 --batch-size 32
```

---

## All Available Flags

| Flag | Options | Default | Purpose |
|------|---------|---------|---------|
| `--verbose` | True/False | True | Console output verbosity |
| `--save-markdown` | flag | off | Save markdown report |
| `--markdown-complexity` | summary/standard/full | full | Report detail level |
| `--save-plots` | flag | off | Save visualizations |
| `--episodes` | integer | 100 | DQN training episodes |
| `--batch-size` | integer | 32 | Experience replay batch size |
| `--update-freq` | integer | 10 | Target network update freq |
| `--test-episodes` | integer | 20 | Test evaluation episodes |

---

## Output Files

```
outputs/
├── reports/
│   └── pricing_analysis_*.md
└── visualizations/
    └── comparison_*.png
```

---

## Complexity Levels Explained

### `--markdown-complexity summary`
- Executive summary only
- Revenue improvement %
- Acceptance rates

### `--markdown-complexity standard`
- Summary metrics
- Detailed comparison
- Price statistics

### `--markdown-complexity full`
- Everything above
- Acceptance rate details
- Statistical summaries
- Performance metrics

---

## Preset Configurations

### Fast (2-3 min)
```bash
python src/pipeline.py --episodes 50 --test-episodes 10 --batch-size 64 --verbose False
```

### Standard (5-7 min)
```bash
python src/pipeline.py --save-markdown --save-plots
```

### Thorough (15-20 min)
```bash
python src/pipeline.py --episodes 300 --batch-size 16 --update-freq 5 --test-episodes 50 --save-markdown --markdown-complexity full --save-plots
```

---

## Example Workflows

### Get Quick Results
```bash
python src/pipeline.py --episodes 75 --save-markdown --markdown-complexity summary
```

### Comprehensive Analysis
```bash
python src/pipeline.py --episodes 200 --batch-size 32 --test-episodes 30 --save-markdown --markdown-complexity full --save-plots
```

### Production-Ready Model
```bash
python src/pipeline.py --episodes 500 --batch-size 32 --update-freq 10 --test-episodes 100 --save-markdown --markdown-complexity full --save-plots
```

---

## Module Functions

### Static Pricing (`modules/static.py`)
```python
from modules.static import (
    calculate_acceptance_rate,
    train_static_model,
    apply_static_pricing,
    get_static_metrics
)
```

### Reinforcement Learning (`modules/reinforcement.py`)
```python
from modules.reinforcement import (
    DQN,
    PricingEnvironment,
    DQNAgent,
    train_dqn_agent,
    test_dqn_agent,
    get_dqn_metrics
)
```

### Pipeline (`pipeline.py`)
```python
from pipeline import (
    main,
    prepare_feature_engineered_data,
    run_static_pricing_pipeline,
    run_dqn_pipeline,
    compare_strategies
)
```

---

## Performance Tips

- **More episodes = Better model quality** but slower
- **Smaller batch size = More updates** but less stable
- **Larger batch size = More stable** but slower convergence
- **Lower update-freq = More target updates** but noisier gradients

---

## Debugging

### No markdown file created?
Use the flag: `--save-markdown`

### Slow execution?
Use: `--episodes 50 --batch-size 64 --verbose False`

### Out of memory?
Use: `--batch-size 16 --test-episodes 5`

---

*For detailed documentation, see `CLI_GUIDE.md`*
