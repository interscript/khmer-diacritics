# interscript-khmer
## Approaches
| Name | Description | CER | Accuracy (%) |
|:----:|:------------|:---:|:------------:|
|Seq2Seq Transformers|This is simple transformers based seq2seq approach that was proposed in the ["Attention is all you need"](https://arxiv.org/pdf/1706.03762.pdf) paper.| **0.3125** | **50.12** |
|Seq2Seq Transformers + Dictionary Lookup | The same as the first approach, but adding Dictionary lookup. Dictionary lookup is an approach of spell correction when we looking for similar (Levenstein distance = 2) words in the Khmer → Transcription dictionary. | 0.4174 | 23.18 |
| Seq2Seq Transformers + SymSpell | The same as the first approach, but adding SymSpell approach for spell correction. The approach is described in [this](https://github.com/wolfgarbe/SymSpell) repository. | - | - |
| Simple mapping | Instead of ML, we use Khmer char to Latin transcriptions dictionary. It was collected via Google Translate and Google Input Tools. Before mapping Khmer to Latin we segment sentences to words via the [khmer-nltk](https://pypi.org/project/khmer-nltk/) module. | 0.42 | 11.00 |
| Seq2Seq Transformers based on mapping data | The same as the first approach, but it's trained on mapping data (described above). So, first of all, we convert Khmer to Latin via the approach above and then the seq2seq transformer is trained. | 0.3387 | 38.85 |

Also, I've made experiments on different Spell correction approaches, but they haven't given any results. For approaches evaluation, synthetic data was created. Synthetic data is data that was created from transcriptions transforms. Approaches that were checked: Seq2Seq Transformers, SymSpell, Dictionary Lookup.

## How to train the model?
```
python python/train.py -i examples/khm-latn/input.csv -t examples/khm-latn/target.csv -en NAME_OF_THE_MODEL
```
This command will run model fitting on default params. If you need custom parameters then check [secryst repo](https://github.com/secryst/secryst).

The best model in ***models/no-punct-no-bug/***.

## How to inference Seq2Seq Transformers model?

```
python python/inference.py -i KHMER_TEXT -en NAME_OF_THE_MODEL
```
If you want to inference the best model, then run:
```
python python/inference.py -i KHMER_TEXT -en no-punct-no-bug
```
