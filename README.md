# Decoders for 🤗 transformers

This package provides a convenient interface for extensible and customizable generation strategies -aka decoders- in 🤗 transformers.

It also provides extra implementations out of the box, like the Stochastic Beam Search decoder.

## Installation
```
pip install decoders
```
## Usage
Simple use of the new interface:
```python
from decoders import inject_supervitamined_decoders
from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained('t5-small')
inject_supervitamined_decoders(model)
model.generate(...)
```
## Decoders

### Stochastic Beam Search
This decoder is a stochastic version of the Beam Search decoder. It is a HF implementation of the paper [Stochastic Beam Search](https://arxiv.org/abs/1903.06059).

It can be used as follows:
```python
from decoders import StochasticBeamSearchDecoder, inject_supervitamined_decoders
from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained('t5-small')
inject_supervitamined_decoders(model)

decoder = StochasticBeamSearchDecoder()
outputs = model.generate(input_ids, generation_strategy=decoder, 
                         num_beams=4, num_return_sequences=4, # sample without repl. = return all beams
                         length_penalty=0.0,  # for correct probabilities, disable length penalty
                         return_dict_in_generate=True, output_scores=True, early_stopping=True,
                         # early stopping because without length penalty, we can discard worse sequences
                         # return_dict_in_generate and output_scores are required for sbs for now,
                         # as scores keep the past generated gumbel noise, which is used by the logits processor
                         )
```

Complete example [here](https://gist.github.com/manueldeprada/839e2446cc4e72dd8eb558c1acbbe85f).