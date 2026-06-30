# Foundation Model Benchmark

Zero-shot time series foundation models benchmarked against tuned statistical and ML baselines on real supply chain demand data.

## Models

### Foundation Models

| Model | Source | Context Length | Notes |
|-------|--------|---------------|-------|
| `TIMESFM25_SL1` | `darts` — `TimesFM2p5Model` | 1× season_length | Short context: 1 year of weekly history, 1 year of monthly history |
| `TIMESFM25_SL2` | `darts` — `TimesFM2p5Model` | 2× season_length | Long context: ~2 years of weekly or monthly history |
| `CHRONOS2_SL2` | `darts` — `Chronos2Model` | 2× season_length | Long context variant of Amazon Chronos 2 |
| `MOIRAI2_SMALL` | `uni2ts` — `Salesforce/moirai-2.0-R-small` | 1000 steps | Moirai 2 small, fixed context window |
| `MOIRAI_MOE_BASE` | `uni2ts` — `Salesforce/moirai-moe-1.0-R-base` | 200 steps | Moirai Mixture-of-Experts base |
| `MOIRAI_MOE_SMALL` | `uni2ts` — `Salesforce/moirai-moe-1.0-R-small` | 200 steps | Moirai Mixture-of-Experts small |

All foundation models are run zero-shot — no fine-tuning on client data.

### ML Baselines

| Model | Source | Objective | Strategy | Estimators | Leaves |
|-------|--------|-----------|----------|-----------|--------|
| `LGBM_BASE_RECURSIVE` | `mlforecast` (Nixtla) | Tweedie (power=1.5) | Recursive | 100 | 90 |
| `LGBM_BASE_DIRECT` | `mlforecast` (Nixtla) | Tweedie (power=1.5) | Direct | 100 | 90 |

Features: lag features, rolling windows, and date encodings derived automatically from data frequency (weekly: lags 1–26, windows [4,13,26,52]; monthly: lags 1–12, windows [3,6,9,12]).

### Statistical Baselines

| Model | Source | Parameters |
|-------|--------|-----------|
| `SES_OPT` | `statsforecast` (Nixtla) — `SimpleExponentialSmoothingOptimized` | α optimized via SSE |
| `SEASONAL_NAIVE` | `statsforecast` (Nixtla) — `SeasonalNaive` | season_length per client |

## Customers

Results cover three anonymized enterprise supply chain clients evaluated on held-out cross-validation windows.

| Customer | Series | Eval Horizon | Profile |
|----------|--------|-------------|---------|
| Customer A | 2,796 | 9-12 weeks ahead | Weekly |
| Customer B | 9,271 | 5 steps ahead | Monthly |
| Customer C | 6,281 | 9-12 periods | Weekly |

## Metrics

- **wMAPE** — weighted mean absolute percentage error (primary accuracy metric)
- **Bias** — signed error; negative = systematic under-forecast
- **wMAPE + |Bias|** — composite rank metric penalizing both error and systematic bias
- **MASE** — mean absolute scaled error (scale-independent cross-client comparison, using Insample Naive Forecast)

## Source Packages

| Package | Used For | Reference |
|---------|----------|-----------|
| [`darts`](https://github.com/unit8co/darts) | TimesFM 2.5, Chronos 2 model wrappers | Unit8 |
| [`sktime`](https://github.com/sktime/sktime) | Foundation model Panel scitype interface | sktime community |
| [`uni2ts`](https://github.com/SalesforceAIResearch/uni2ts) | Moirai / Moirai-MoE inference | Salesforce AI Research |
| [`statsforecast`](https://github.com/Nixtla/statsforecast) | SES_OPT, Seasonal Naive | Nixtla |
| [`mlforecast`](https://github.com/Nixtla/mlforecast) | LGBM recursive and direct models | Nixtla |
| [`lightgbm`](https://github.com/microsoft/LightGBM) | Gradient boosting backend | Microsoft |
