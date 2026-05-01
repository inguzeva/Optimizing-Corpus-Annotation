# Corpus Source Notes

## Source: Altai poem
- Title: `ЧӦЛДӦРДИН ЭЭЗИ – МЕН`
- Author: **Лазарь Васильевич Кокышев**
- Added from user-provided scanned images.
- Files populated:
  - `corpora/input/corpus.txt`
  - `corpora/input/corpus.jsonl`

## Source: Hugging Face parallel corpus
- Dataset: `Lil-Graver/small-altai-corpus`
- URL: `https://huggingface.co/datasets/Lil-Graver/small-altai-corpus`
- License: `CC-BY-4.0`
- Structure used in this project:
  - `Алтайский` column is imported as corpus text
  - `Русский` column is stored as `translation_ru` metadata in JSONL
- Import script:
  - `scripts/75_import_hf_small_altai.py`
- Files populated:
  - `corpora/input/small_altai_corpus.jsonl`
  - `corpora/input/corpus.txt`
  - `corpora/input/corpus.jsonl`
