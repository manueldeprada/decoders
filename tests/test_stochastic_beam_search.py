from decoders import inject_supervitamined_decoders, SmallProbTransformer, SmallProbTransformerConfig, \
    BinaryCodeTransformer, StochasticBeamSearchDecoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def test_simple_bs_single():
    inputs = ["translate English to German: How old are you?"]
    _test_simple_sbs(inputs)


def test_simple_bs_batch():
    inputs = ["translate English to German: What is your name, my dear Friend? I missed you so much",
        "translate English to German: How old are you?",
        "a b c 1 2 ",
        "summarize: Lorem ipsum dolor "
    ]
    _test_simple_sbs(inputs)


def _test_simple_sbs(inputs):
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    model.eval()
    inject_supervitamined_decoders(model)
    inputs = tokenizer(inputs,
                       return_tensors="pt", padding=True, truncation=True
                       )
    outputs = model.generate(**inputs,
                             generation_strategy=StochasticBeamSearchDecoder(),
                             max_new_tokens=100, num_beams=5, num_return_sequences=5,
                             )

    print(f"generated text: {tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)}")
    print(f"generated probs: {outputs.sequences_scores}")
    print(f"generated gumbels: {outputs.last_scores}")


def test_binary_transformer():
    import torch
    inputs = torch.tensor([[-1]] * 2)
    model = BinaryCodeTransformer(n=6)
    inject_supervitamined_decoders(model)
    output = model.generate(inputs,
                            generation_strategy=StochasticBeamSearchDecoder(),
                            num_beams=100,
                            num_return_sequences=100,
                            max_new_tokens=100,
                            )
    print(f"generated seqs: {output.sequences}")
    print(f"generated probs: {output.sequences_scores}")
    print(f"generated gumbels: {output.last_scores}")


def test_small_prob_transformer():
    import torch
    torch.manual_seed(43)
    inputs = torch.tensor([[-2]] * 2)
    model = SmallProbTransformer(SmallProbTransformerConfig())
    inject_supervitamined_decoders(model)
    output = model.generate(inputs,
                            generation_strategy=StochasticBeamSearchDecoder(),
                            max_new_tokens=200, num_beams=100, num_return_sequences=100,
                            eval_by_score=True,
                            )
    print(f"generated seqs: {output.sequences}")
    print(f"generated gumbels: {output.last_scores}")
    print(f"generated probs: {output.sequences_scores}")
    routes = output.sequences[:, 1]
    print(f"routes: {routes}")
    print(f"bins: {routes.bincount(minlength=9)}")


if __name__ == '__main__':
    import sys

    def debugger_is_active() -> bool:
        """Return if the debugger is currently active"""
        return hasattr(sys, 'gettrace') and sys.gettrace() is not None

    if debugger_is_active():
        # test_binary_transformer()
        test_small_prob_transformer()
    else:
        from arsenal import testing_framework
        testing_framework(globals())
