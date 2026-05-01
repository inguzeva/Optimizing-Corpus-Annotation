# Optimizing Corpus Annotation for Low-Resource Languages Using Multimodal Data and Machine Learning

Pipeline to convert a large dictionary PDF into structured CSVs and use them for weak labeling, self-training, and QA via a small web app.

Now includes a simple multimodal weak-labeling mode:
- `text-only` confidence
- `text+pdf` confidence (text score + PDF/layout quality features)

## Key locations
- Dictionary PDF: project root (see config.yaml)
- Parsed CSVs: data/processed/parsed/
- Clean CSVs (source of truth for ML): data/processed/clean/
- Raw/interim artifacts: data/raw/ and data/interim/
- Web app: app/

## Typical flow (high level)
1) Parse PDF to raw/interim artifacts
2) Build parsed CSVs
3) Clean parsed CSVs
4) Build SQLite for web QA
5) Run weak labeling and training

See scripts/ for step-by-step utilities.

## Weak labeling modes
- `python scripts/70_weak_annotate_corpus.py --mode text-only`
- `python scripts/70_weak_annotate_corpus.py --mode text+pdf --pdf-weight 0.30`

## Import additional HF corpus
- Import and merge `Lil-Graver/small-altai-corpus` into the main corpus files:
  - `python scripts/75_import_hf_small_altai.py`
- Main corpus path is configured in `config.yaml` and now points to the merged corpus.

## Independent gold set (50-80 sentences, no weak_labels input)
- Build independent gold from corpus + dictionary source-of-truth:
  - `python scripts/92_build_independent_gold.py --corpus corpora/input/corpus.txt --out corpora/gold/gold_independent_65.jsonl`
- Optional manual correction layer (applied after bootstrap):
  - `corpora/gold/manual_overrides.jsonl`

## External baseline for ML contribution
- Generate predictions from an external HF token-classification model:
  - `python scripts/86_predict_external_baseline.py --input corpora/gold/gold_independent_65.jsonl --output corpora/weak_labels/external_baseline_predictions.jsonl`
- If transformers stack is not installed yet, run:
  - `pip install -r requirements/ml.txt`

## Unified evaluation report
- Compare weak modes + local ML model + external baseline on independent gold:
  - `python scripts/95_compare_modes.py --gold corpora/gold/gold_independent_65.jsonl --out reports/mode_comparison.json`
