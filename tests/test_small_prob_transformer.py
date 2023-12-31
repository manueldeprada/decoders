import torch
from decoders import inject_supervitamined_decoders, SmallProbTransformer, SmallProbTransformerConfig, SamplingDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = SmallProbTransformerConfig()
model = SmallProbTransformer(config).to(device)
inject_supervitamined_decoders(model)


def test_ancestral_sample():
    torch.manual_seed(43)
    input_ids = torch.tensor([[-1]]).to(device)
    output = model.generate(input_ids,
                            generation_strategy=SamplingDecoder(),
                            max_new_tokens=200, top_k=0, do_sample=True,
                            num_return_sequences=20, )
    routes = output.sequences[:, 1]
    bins = routes.bincount(minlength=config.real_vocab_size)
    print(f"generated tokens: {output.sequences}")
    print(f"routes: {routes}")
    print(f"bins: {bins}")
    print(f"generated probs: {output.sequences_scores}")


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())