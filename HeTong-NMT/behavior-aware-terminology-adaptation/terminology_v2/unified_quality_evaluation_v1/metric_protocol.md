# Unified Translation Quality Metric Protocol

Evaluator source: user-provided `evaluate_gemma_e4b.py`, reproduced in `unified_evaluator.py`.

## BLEU

SacreBLEU `BLEU(tokenize="none", effective_order=True).sentence_score()` is run once per sample, then arithmetic-mean aggregated. Tokenization is selected from the reference language: `jieba.cut(clean(text))` for Chinese; `TIBETAN_SEPARATORS = r"[\s་།༎༏༐༑༔]+"` for Tibetan; whitespace tokens otherwise.

## chrF++

SacreBLEU `CHRF(word_order=2).sentence_score()` is run once per sample, then arithmetic-mean aggregated.

## ROUGE-L

`rouge_score.rouge_scorer.RougeScorer(["rougeL"], tokenizer=LangAwareTokenizer())` computes per-sample `fmeasure`, then arithmetic-mean aggregates it. The tokenizer is the same language-aware Chinese/Tibetan tokenizer described for BLEU.

## Retained Metrics and Integrity

`strict_term_pass` and `canonical_term_pass` are read directly from existing prediction JSONLs; no checker is rerun. Every source JSONL has SHA-256 recorded before and after reading, and every available method is required to contain exactly 2,000 unique record IDs per direction.
