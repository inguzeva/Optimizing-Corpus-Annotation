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
