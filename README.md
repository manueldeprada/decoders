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
Note that when sampling without replacement, you must set `num_beams` and `num_return_sequences` to the same value, the number of SWOR samples that you want to obtain.

Of course, the samples for the same input are not independent. If you want R different groups of SWOR samples of size n, you should replicate your batched input tensor by R, and then set num_beams and num_return_sequences to n.

See [here](https://gist.github.com/manueldeprada/839e2446cc4e72dd8eb558c1acbbe85f) for a full example.

## Included goodies
### BinaryCodeTransformer

The BinaryCodeTransformer is a custom transformer model that acts like a probabilistic binary sequence generator. Given a discrete probability distribution over all possible binary sequences of a given length, it generates a sequence of that length according to that distribution. It is useful to test HF compatible sample-without-replacement decoders, like the Stochastic Beam Search decoder.

The code maps each of the 2^n possible binary sequences of length n to its positive integer decimal representation. Then, it uses that number as the index of the corresponding probability in the input distribution. Since we are interested in autoregressive generation, the model computes the conditional probabilities by summing over the possible continuations of the sequence.

### FakeTransformer

The FakeTransformer operates as a very simple Probabilistic Finite State Automaton. See [here](https://manueldeprada.com/blog/posts/toy-probabilistic-transformer/) for a full explanation.

