import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, NoBadWordsLogitsProcessor
from decoders import inject_supervitamined_decoders, LogitsProcessorList
import time


def test_simple_bs_single():
    inputs = ["translate English to German: How old are you?"]
    _test_simple_beam_search(inputs)


def test_simple_bs_batch():
    inputs = ["translate English to German: What is your name, my dear Friend? I missed you so much",
        "translate English to German: How old are you?",
        "a b c 1 2 ",
        "summarize: Lorem ipsum dolor "
    ]
    _test_simple_beam_search(inputs)


def _test_simple_beam_search(inputs):
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    inject_supervitamined_decoders(model)
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    t0 = time.time()
    outputs = model.generate(**inputs,
                             generation_strategy="beam_search",
                             do_sample=False,
                             max_new_tokens=100, num_beams=5,
                             return_dict_in_generate=True,
                             )
    t1 = time.time()
    gold_seqs = model.generate(**inputs,
                               num_beams=5, do_sample=False, max_new_tokens=100, length_penalty=0.0,
                               early_stopping=True, return_dict_in_generate=True, output_scores=True)
    t2 = time.time()
    print(f"generated text: {tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)}")
    print(f"gold text: {tokenizer.batch_decode(gold_seqs.sequences, skip_special_tokens=True)}")
    print(f"generated probs: {outputs.sequences_scores}")
    print(f"gold probs: {gold_seqs.sequences_scores}")
    print(f"simple took {t1 - t0:.2f}s, gold took {t2 - t1:.2f}s")
    assert torch.all(
        outputs.sequences == gold_seqs.sequences), f"output: {outputs.sequences}, gold: {gold_seqs.sequences}"


def test_logit_processor():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    inject_supervitamined_decoders(model)
    inputs = tokenizer(["translate English to German: How old are you?", ],
                       return_tensors="pt", padding=True, truncation=True
                       )
    outputs = model.generate(**inputs,
                             generation_strategy="beam_search",
                             max_new_tokens=100, num_beams=5, do_sample=False,
                             logits_processor=LogitsProcessorList(
                                 [NoBadWordsLogitsProcessor(bad_words_ids=[[292]], eos_token_id=1)]),
                             return_dict_in_generate=True,
                             )
    print(f"generated logp: {outputs.sequences_scores}")
    print(f"generated tokens: {outputs.sequences}")
    print(f"generated text: {tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)}")
    assert torch.all(outputs.sequences == torch.tensor([[0, 2739, 4445, 15840, 146, 58, 1]]))

if __name__ == '__main__':
    import sys

    def debugger_is_active() -> bool:
        """Return if the debugger is currently active"""
        return hasattr(sys, 'gettrace') and sys.gettrace() is not None

    if debugger_is_active():
        # test_binary_transformer()
        test_simple_bs_single()
    else:
        from arsenal import testing_framework
        testing_framework(globals())