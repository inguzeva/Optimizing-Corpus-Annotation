# Independent Gold Guidelines

1. Gold labels are created without reading `corpora/weak_labels/*`.
2. Source data for sampling is `corpora/input/corpus.txt` (or `.jsonl` corpus with the same text content).
3. Initial labels are bootstrapped from dictionary source-of-truth (`data/processed/clean/*` + `dopslovar.txt`) and then manually corrected.
4. Manual corrections are stored in `corpora/gold/manual_overrides.jsonl`.
5. Final dataset is exported as JSONL in this shape:
   - `{"sent_id": "...", "text": "...", "tokens": [{"token_idx": 0, "token": "...", "label": "O|LEX"}]}`
6. Target size for independent evaluation subset: 50-80 sentences.
