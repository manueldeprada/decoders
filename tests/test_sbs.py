# test_decoders.py
import pytest
from decoders import inject_supervitamined_decoders, StochasticBeamSearchDecoder, FakeTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


# Assumes that you've installed the necessary packages and 'decoders' module is available
# You might need to mock or adapt the dependencies if they are not straightforward to install


@pytest.fixture(scope="module")
def fake_transformer_model():
    torch.manual_seed(0)
    model = FakeTransformer()
    inject_supervitamined_decoders(model)
    return model


@pytest.fixture(scope="module")
def t5_model_and_tokenizer():
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)
    inject_supervitamined_decoders(model)
    return model, tokenizer


def test_fake_transformer(fake_transformer_model):
    decoder = StochasticBeamSearchDecoder()

    result = fake_transformer_model.generate(
        input_ids=torch.tensor([[0], [0]]),  # 0 is BOS in the FakeTransformer
        generation_strategy=decoder,
        num_return_sequences=3,  # sample without replacement = return all beams
        num_beams=3,
        length_penalty=0.0,  # for correct probabilities, disable length penalty
        return_dict_in_generate=True, output_scores=True, early_stopping=True,
    )

    assert torch.equal(result.sequences, torch.tensor([[0, 1, 2, 3],
                                                       [0, 2, 2, 3],
                                                       [0, 1, 1, 3],
                                                       [0, 1, 2, 3],
                                                       [0, 2, 2, 3],
                                                       [0, 2, 1, 3]]))
    assert (result.sequences_scores.exp() - torch.tensor(
        [0.6750, 0.2250, 0.0750, 0.6750, 0.2250, 0.0250])).abs().max() < 1e-4


def test_t5(t5_model_and_tokenizer):
    model, tokenizer = t5_model_and_tokenizer
    input_ids = tokenizer(["translate to german: hi!"], padding=True, return_tensors="pt").input_ids

    # example of using greedy search and beam search with the new proposed decoder interface
    _ = model.generate(input_ids=input_ids, generation_strategy='greedy', max_length=10)
    _ = model.generate(input_ids=input_ids, generation_strategy='beam_search', max_length=10, num_beams=5)

    # stochastic beam search with the new interface and t5
    decoder = StochasticBeamSearchDecoder()
    outputs = model.generate(input_ids=input_ids, generation_strategy=decoder, max_length=10,
                             num_beams=5, num_return_sequences=5,
                             output_scores=True, return_dict_in_generate=True)
    
    decoded_outputs = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    assert len(decoded_outputs) == 5  # You might want to add more specific assertions here
    # For example, checking that the decoded outputs are valid translations
