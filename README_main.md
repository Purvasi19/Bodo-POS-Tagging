# Bodo POS Tagging

Part-of-speech tagging for the Bodo (Boro) language. This repo contains
two independent implementations trained on the same POS-tagged corpus:

1. **Fine-tuned IndicBERTv3** — a transformer-based tagger, fine-tuned
   for token classification.
2. **BiLSTM+CRF** — a sequence-labeling model using bidirectional LSTM
   layers with a CRF decoding layer.

Both models are trained and evaluated on the same dataset, so their
results are directly comparable.

## Dataset

A manually/semi-automatically POS-annotated Bodo corpus:

| | |
|---|---|
| Sentences | 6,005 |
| Tokens | 90,560 |
| Unique fine-grained tags | 35 |
| Format | space-separated `word\TAG` tokens, one sentence per row |

The tagset follows the standard Indian-language (BIS) POS scheme —
`N` (noun), `V` (verb), `JJ` (adjective), `PSP` (postposition), `CC`
(conjunction), `QT` (quantifier), `DM` (demonstrative), `PR` (pronoun),
`RP` (particle), `RD` (residual: punctuation, symbols, foreign words),
with fine-grained sub-tags such as `N_NNP`, `V_VM`, `RD_PUNC`, etc.

Full dataset documentation, tag distribution, and a parsing snippet are
in [`data/README.md`](data/README.md).

## Project structure

```
.
├── data/
│   ├── pos_tagged.xlsx        # raw annotated corpus
│   └── README.md              # dataset format, tagset, stats
├── indicbert/
│   ├── train.py                # fine-tunes IndicBERTv3 for token classification
│   ├── evaluate.py
│   └── predict.py
├── bilstm_crf/
│   ├── model.py                 # BiLSTM+CRF architecture
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Status

| Component | Status |
|---|---|
| Dataset preparation | ✅ Done |
| BiLSTM+CRF training | ✅ Trained |
| IndicBERTv3 fine-tuning | ✅ Trained |
| Evaluation / comparison | 🚧 In progress |

Both models have been trained on the full dataset. A side-by-side
evaluation (accuracy, per-tag F1, confusion analysis) is in progress —
see [Results](#results) below.

## Setup

```bash
git clone <your-repo-url>
cd bodo-pos-tagging
pip install -r requirements.txt
```

## Data preparation

Both models expect `(token, tag)` sequences per sentence. Convert the raw
corpus with:

```python
import pandas as pd

df = pd.read_excel("data/pos_tagged.xlsx", header=None)
sentences = df[0].tolist()[1:]  # skip header row

def parse_sentence(sentence: str):
    pairs = []
    for token in sentence.split():
        word, _, tag = token.rpartition("\\")
        pairs.append((word, tag))
    return pairs

tagged_sentences = [parse_sentence(s) for s in sentences]
```

Split into train/dev/test sets before training either model (e.g. an
80/10/10 split), and keep the split consistent across both models so
results are comparable.

## Model 1: IndicBERTv3 (fine-tuned)

Token classification head on top of IndicBERTv3 embeddings.

```bash
python indicbert/train.py \
  --data data/pos_tagged.xlsx \
  --output_dir checkpoints/indicbert \
  --epochs 10 \
  --batch_size 16 \
  --lr 2e-5
```

```bash
python indicbert/evaluate.py \
  --checkpoint checkpoints/indicbert \
  --test_data data/test.json
```

```bash
python indicbert/predict.py \
  --checkpoint checkpoints/indicbert \
  --text "थेवबो मायथायाव जन असम"
```

## Model 2: BiLSTM+CRF

A BiLSTM encoder over token embeddings, with a CRF layer for structured
tag decoding.

```bash
python bilstm_crf/train.py \
  --data data/pos_tagged.xlsx \
  --output_dir checkpoints/bilstm_crf \
  --epochs 30 \
  --batch_size 32 \
  --hidden_dim 256 \
  --embedding_dim 100
```

```bash
python bilstm_crf/evaluate.py \
  --checkpoint checkpoints/bilstm_crf \
  --test_data data/test.json
```

```bash
python bilstm_crf/predict.py \
  --checkpoint checkpoints/bilstm_crf \
  --text "थेवबो मायथायाव जन असम"
```

## Results

| Model | Accuracy | 
|---|---|
| IndicBERTv3 (fine-tuned) | 79 |
| BiLSTM+CRF | 73 | 

*(Fill in once the evaluation script has been run on a held-out test
split — see Next steps.)*

## Next steps

- [ ] Finalize train/dev/test split and keep it fixed across both models
- [ ] Run full evaluation (accuracy, per-tag precision/recall/F1) for both models
- [ ] Compare error patterns between the transformer and BiLSTM+CRF approaches
- [ ] Handle rare compound tags (`V_VM_VNF`, `V_VM_VF`, `V_VAUX_VF`) consistently
- [ ] Release model checkpoints / inference demo

## Citation / acknowledgments

If you use this dataset or these models, please cite this repository and
credit the original corpus annotators.
