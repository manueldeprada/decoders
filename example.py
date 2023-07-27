from decoders import inject_supervitamined_decoders, StochasticBeamSearchDecoder, FakeTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


# pip install decoders

# this demonstration uses a fake toy transformer (https://manueldeprada.com/blog/posts/toy-probabilistic-transformer/)
# to test the correctness of the stochastic beam search implementation
def test_fake_transformer():
    torch.manual_seed(0)

    model = FakeTransformer()

    inject_supervitamined_decoders(model)

    decoder = StochasticBeamSearchDecoder()

    result = model.generate(input_ids=torch.tensor([[0], [0]]),  # 0 is BOS in the FakeTransformer
                            generation_strategy=decoder,
                            num_return_sequences=3,  # sample without replacement = return all beams
                            num_beams=3,
                            length_penalty=0.0,  # for correct probabilities, disable length penalty
                            return_dict_in_generate=True, output_scores=True, early_stopping=True,
                            # early stopping bc without length penalty, we can discard worse sequences
                            # return_dict_in_generate and output_scores are required for sbs for now,
                            # as scores keep the past generated gumbel noise, which is used by the logits processor
                            )
    assert (result.sequences == torch.tensor([[0, 1, 2, 3],
                                              [0, 2, 2, 3],
                                              [0, 1, 1, 3],
                                              [0, 1, 2, 3],
                                              [0, 2, 2, 3],
                                              [0, 2, 1, 3]])).all()
    assert (result.sequences_scores.exp() - torch.tensor(
        [0.6750, 0.2250, 0.0750, 0.6750, 0.2250, 0.0250])).abs().max() < 1e-4
    print("Test for sbs passed")


# this example uses t5 to demonstrate the usage of the new interface and the stochastic beam search decoder
def test_t5():
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)
    input_ids = tokenizer(["translate to german: hi!"], padding=True, return_tensors="pt").input_ids

    # the magic trick of decoders!
    inject_supervitamined_decoders(model)

    # example of using greedy search and beam search with the new proposed decoder interface
    outputs = model.generate(input_ids=input_ids, generation_strategy='greedy', max_length=10)
    outputs = model.generate(input_ids=input_ids, generation_strategy='beam_search', max_length=10, num_beams=5)

    # stochastic beam search with the new interface and t5
    decoder = StochasticBeamSearchDecoder()
    outputs = model.generate(input_ids=input_ids, generation_strategy=decoder, max_length=10,
                             num_beams=5, num_return_sequences=5,
                             output_scores=True, return_dict_in_generate=True)
    print("Sample without replacement from T5:")
    print(tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True))


if __name__ == '__main__':
    test_fake_transformer()
    test_t5()
