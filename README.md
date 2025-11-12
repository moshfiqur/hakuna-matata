# Business Intelligence Utilities

This repository packages a small toolkit that automates three common steps in BI prototyping:

1. **Dataset acquisition** from Kaggle into a structured `data/` directory.
2. **Column semantics extraction** that inspects a CSV, classifies each column with a zero-shot language model, and stores the results in `output/column_semantics/`.
3. **Semantic mapping** across datasets that uses transformer embeddings and cosine similarity to find related columns and exports the relationships to `output/semantic_mappings/`.

The scripts are designed to run independently so you can pull down new data, describe it, and align it with other datasets without leaving the command line.

## Repository Structure

```
src/
├── data_loader.py         # Kaggle downloader and loader utilities
├── semantic_extractor.py  # Zero-shot column semantics generator
└── semantic_mapper.py     # Cross-dataset similarity mapper
```

Other folders (`data/`, `output/`, `notebooks/`) contain downloaded datasets, generated artifacts, or exploratory work.

## Script Details

### `src/data_loader.py`

The `EnterpriseDataLoader` wraps the Kaggle API to fetch datasets listed in a user-provided manifest. Each Kaggle dataset is downloaded into its own folder under `data/<kaggle_ref>/`, leaving original file names intact. You can call `get_enterprise_datasets()` with a list of tuples `(kaggle_ref, dataset_name)` to retrieve the dataframes you need.

Key points:
- Authenticates with the Kaggle API when the loader is instantiated, so make sure you have your Kaggle credentials configured locally.
- Skips re-downloading when CSV files already exist in the dataset folder to avoid redundant transfers.
- Returns pandas DataFrames so downstream scripts can inspect or export them immediately.

Usage snippet:
```bash
python -c "from src.data_loader import EnterpriseDataLoader;\nloader = EnterpriseDataLoader();\nprint(loader.get_enterprise_datasets([('rohitsahoo/sales-forecasting','superstore_sales')]).keys())"
```

### `src/semantic_extractor.py`

This script reads a CSV, infers column-level metadata (raw dtype, normalized dtype, sample values), and classifies each column’s semantic category with a **zero-shot transformer pipeline**:
- **Zero-shot pipeline**: Uses `transformers.pipeline('zero-shot-classification')` with the lightweight `valhalla/distilbart-mnli-12-1` model. Zero-shot means the model can decide among custom labels (**financial**, **compensation**, etc.) without being fine-tuned specifically for this task. Labels and their descriptive prompts can be overridden via `--label-config`.
- **Heuristic fallback**: If the confidence score is below a threshold (configurable via `--confidence-threshold`), or if inference fails, the script falls back to quick heuristics based on column names and data types.
- **Normalized dtypes**: In addition to the raw pandas dtype (which is often `object`), the extractor emits a `normalized_dtype` column (string, integer, float, boolean, datetime, etc.) for easier consumption in BI tools.

Example command:
```bash
python src/semantic_extractor.py path/to/data.csv \
    --dataset-name my_dataset \
    --output-dir output/column_semantics \
    --model-name valhalla/distilbart-mnli-12-1 \
    --label-config configs/column_labels.yaml \
    --confidence-threshold 0.5
```

The output CSV contains one row per column with the inferred category, confidence score, samples (stored as JSON), and dtype information.

### `src/semantic_mapper.py`

The mapper consumes every semantics CSV in `output/column_semantics/`, loads them as DataFrames, and computes **semantic similarity** between column names across dataset pairs.

Technical concepts used here:
- **Transformer embeddings**: The script uses the same Hugging Face tokenizer/model pair as the original `EnterpriseSemanticExtractor` to embed column names via mean-pooled hidden states. Each column name becomes a vector in a high-dimensional semantic space.
- **Cosine similarity**: After embedding, the script grades how similar two column vectors are with cosine similarity (`sklearn.metrics.pairwise.cosine_similarity`). Scores close to 1.0 mean the columns likely describe the same thing.
- **Thresholding & mapping**: For every pair of datasets, it keeps column pairs whose similarity is above `--threshold` (default 0.75) and writes the matches to `output/semantic_mappings/<datasetA>__vs__<datasetB>.csv`.

Command example:
```bash
python src/semantic_mapper.py output/column_semantics \
    --output-dir output/semantic_mappings \
    --model-name bert-base-uncased \
    --threshold 0.8
```

Each mapping file lists dataset names, columns, semantic categories (carried over from the extractor), and similarity scores so you can quickly align schema between sources.

## Technical Glossary

| Term | Explanation | Used in |
|------|-------------|---------|
| Zero-shot pipeline | A Hugging Face `transformers` pipeline that can classify text into arbitrary labels without task-specific training, driven by natural-language prompts. | `semantic_extractor.py` |
| Transformer embeddings | Dense vector representations produced by a transformer model (BERT, DistilBART) that capture semantic meaning of column names. | `semantic_mapper.py` |
| Cosine similarity | A metric that measures the cosine of the angle between two vectors; ranges from -1 to 1 and signals how close two embeddings are. | `semantic_mapper.py` |
| Sample values | Representative values sampled from each column to provide context for classifiers and downstream review. | `semantic_extractor.py` |
| Normalized dtype | Human-friendly data type derived from pandas dtype (e.g., `object` → `string`). | `semantic_extractor.py` |

## Getting Started

1. Install dependencies (ideally in a virtualenv):
   ```bash
   pip install -r requirements.txt
   ```
2. Make sure Kaggle credentials exist in `~/.kaggle/kaggle.json` if you intend to download datasets.
3. Run the scripts in order:
   1. `src/data_loader.py` (optional CLI or import) to populate `data/`.
   2. `src/semantic_extractor.py` for each CSV you want described.
   3. `src/semantic_mapper.py` once you have multiple semantics files.

Feel free to extend the label prompts, swap in different transformer models, or plug the outputs into notebooks/dashboards for richer BI insights.
