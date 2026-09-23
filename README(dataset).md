# Bodo POS-Tagged Corpus

A manually/semi-automatically POS-annotated corpus of Bodo (Boro) text,
used to train and evaluate the POS taggers in this repository
(fine-tuned **IndicBERTv3** and **BiLSTM+CRF**).

## Dataset summary

| | |
|---|---|
| Sentences | 6,005 |
| Tokens | 90,560 |
| Unique fine-grained tags | 35 |
| Format | 1 sentence per row, space-separated `word\TAG` tokens |
| File | `1790185857778_pos_tagged.xlsx` (single column, header `Value`) |

## Format

Each cell contains one full sentence. Tokens are separated by spaces, and
each token is annotated as:

```
<word>\<TAG>
```

Example (row from the corpus):

```
थेवबो\CC_CCS 1658\N_NN मायथायाव\N_NN जन\N_NNP असम\N_NNP ... ।\RD_PUNC
```

Sub-tagged categories use an underscore, e.g. `N_NNP` (proper noun) or
`V_VM` (main verb); a few coarse tags appear without a sub-tag (e.g. `JJ`,
`PSP`, `RB`).

## Tagset

The corpus follows the standard Indian-language (BIS) POS tagging
scheme. Coarse-grained tag distribution across the corpus:

| Tag | Description | Token count |
|---|---|---|
| N | Noun | 36,423 |
| V | Verb | 17,519 |
| RD | Residual (punctuation, foreign words, symbols, unknown) | 15,906 |
| JJ | Adjective | 5,351 |
| QT | Quantifier | 3,647 |
| CC | Conjunction | 2,936 |
| DM | Demonstrative | 2,597 |
| PR | Pronoun | 2,144 |
| PSP | Postposition | 1,945 |
| RB | Adverb | 1,788 |
| RP | Particle | 304 |

Common fine-grained tags include `N_NN` (common noun), `N_NNP` (proper
noun), `N_NST` (spatial/temporal noun), `V_VM` (main verb), `V_VAUX`
(auxiliary verb), `RD_PUNC` (punctuation), `RD_RDF` (foreign word,
mostly English text embedded in the corpus), `CC_CCD`/`CC_CCS`
(coordinating/subordinating conjunction), `QT_QTC`/`QT_QTO`/`QT_QTF`
(cardinal/ordinal/general quantifiers), `PR_PRP`/`PR_PRF`/`PR_PRI`
(personal/reflexive/interrogative pronouns), and `DM_DMD`/`DM_DMI`/`DM_DMQ`/`DM_DMR`
(demonstrative subtypes).

## Loading the data

```python
import pandas as pd

df = pd.read_excel("1790185857778_pos_tagged.xlsx", header=None)
sentences = df[0].tolist()[1:]  # skip header row ("Value")

# Parse one sentence into (word, tag) pairs
def parse_sentence(sentence: str):
    pairs = []
    for token in sentence.split():
        word, _, tag = token.rpartition("\\")
        pairs.append((word, tag))
    return pairs

print(parse_sentence(sentences[0]))
```

## Notes / known quirks

- A handful of tokens carry compound tags (`V_VM_VNF`, `V_VM_VF`,
  `V_VAUX_VF`) — these are rare (≤8 occurrences each) and worth checking
  before training, since they may need to be normalized to their base
  category depending on what the tagger expects.
- Some sentences embed English words/phrases (tagged `RD_RDF`), typically
  book/author citations — these are real tokens in the corpus, not noise.
- No malformed tokens were found (every token successfully splits into
  `word` + `tag`).

## Using this corpus with the models in this repo

Both the IndicBERTv3 and BiLSTM+CRF taggers expect `(token, tag)`
sequences per sentence — use `parse_sentence()` above to convert each row
into that format before building your train/dev/test splits.
