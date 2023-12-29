from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

from decoders import inject_supervitamined_decoders
from decoders.simple.ancestral import SamplingDecoder
from decoders.toolbox import compute_logprobs_from_scores, compute_true_logprobs

def test_ancestral_noscores():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    inject_supervitamined_decoders(model)
    inputs = tokenizer(["translate English to German: What is your name, my dear Friend? I missed you so much",
                           "translate English to German: How old are you?",
                       ],
                       return_tensors="pt", padding=True, truncation=True
                       )
    outputs = model.generate(**inputs, generation_strategy=SamplingDecoder(),
                             generation_config=GenerationConfig(max_new_tokens=100, top_k=0,
                                                                do_sample=True, output_scores=False))
    # num_beams=1, do_sample=1, top_k=0,
    # return_dict_in_generate=True, output_scores=True)
    gen_logp = outputs.sequences_scores
    print(f"generated text: {tokenizer.batch_decode(outputs.sequences)}")
    print(f"generated tokens: {outputs.sequences}")
    print(f"generated logp: {gen_logp}")
    assert outputs.scores is None
    _, recompute_logp = compute_true_logprobs(model, outputs.sequences, encoder_input=inputs)
    print(f"true logp: {recompute_logp.sum(dim=1)}")

def test_ancestral():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    inject_supervitamined_decoders(model)
    inputs = tokenizer(["translate English to German: What is your name, my dear Friend? I missed you so much",
                           "translate English to German: How old are you?",
                       ],
                       return_tensors="pt", padding=True, truncation=True
                       )
    outputs = model.generate(**inputs, generation_strategy=SamplingDecoder(),
                             generation_config=GenerationConfig(max_new_tokens=100, top_k=0,
                                                                do_sample=True, output_scores=True))
    # num_beams=1, do_sample=1, top_k=0,
    # return_dict_in_generate=True, output_scores=True)
    gen_logp = outputs.sequences_scores
    print(f"generated text: {tokenizer.batch_decode(outputs.sequences)}")
    print(f"generated tokens: {outputs.sequences}")
    print(f"generated logp: {gen_logp}")
    logp_computed = compute_logprobs_from_scores(model, outputs)
    print(f"compute_transition logp: {logp_computed.sum(dim=1)}")
    _, recompute_logp = compute_true_logprobs(model, outputs.sequences, encoder_input=inputs)
    print(f"true logp: {recompute_logp.sum(dim=1)}")

if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())