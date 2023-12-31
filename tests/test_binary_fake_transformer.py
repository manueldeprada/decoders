import torch
from decoders import BinaryCodeTransformer, BeamSearchDecoder, inject_supervitamined_decoders, \
    batched_bin_vec_to_decimal, FakeTransformer, StochasticBeamSearchDecoder


def test_fake_transformer():
    torch.manual_seed(42)
    model = FakeTransformer()
    inject_supervitamined_decoders(model)
    result = model.generate(input_ids=torch.tensor([[0], [0]]),
                            generation_strategy=StochasticBeamSearchDecoder(),
                            num_return_sequences=3,
                            num_beams=3,
                            )
    print(f"generated seqs: {result.sequences}")
    print(f"generated probs: {result.sequences_scores.exp()}")
    assert torch.all(result.sequences == torch.tensor([[0, 1, 2, 3],
                                                          [0, 2, 2, 3],
                                                          [0, 1, 1, 3],
                                                          [0, 2, 2, 3],
                                                          [0, 1, 1, 3],
                                                          [0, 1, 2, 3]]))
    assert torch.allclose(result.sequences_scores.exp(), torch.tensor([0.6750, 0.2250, 0.0750, 0.2250, 0.0750, 0.6750]))


def test_binary_code_transformer_scores():
    torch.manual_seed(0)
    model = BinaryCodeTransformer()
    inject_supervitamined_decoders(model)
    input_ids = torch.tensor([[-1]] * 1000)
    output = model.generate(input_ids, generation_strategy=BeamSearchDecoder())
    numbers = batched_bin_vec_to_decimal(output.sequences[:, 1:])
    analytical_scores = model.p[numbers].log()
    assert torch.allclose(output.sequences_scores.float(), analytical_scores.float())


if __name__ == "__main__":
    from arsenal import testing_framework

    testing_framework(globals())
