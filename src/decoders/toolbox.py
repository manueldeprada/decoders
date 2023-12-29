import torch

def _prepare_decoder_input(model, sequences):
    batch_size = sequences.shape[0]
    device = sequences.device
    start_token = model.config.decoder_start_token_id
    assert torch.all(sequences[:, 0] == start_token)
    decoder_input = torch.tensor([[start_token]], dtype=torch.long, device=device).repeat(batch_size, 1)
    return decoder_input.view(batch_size, 1)


def compute_transition_scores_nonbeam(sequences, scores, normalize_logits=True):
    if len(sequences.shape) == 1:
        sequences = sequences.unsqueeze(0)
    scores = torch.stack(scores, dim=1) # batch_size x seq_len x vocab_size
    if normalize_logits:
        scores = scores.log_softmax(dim=-1)
    sequences = sequences[:, 1:]
    transition_scores = scores.gather(dim=-1, index=sequences.unsqueeze(-1)).squeeze(-1)
    return transition_scores


def compute_true_logprobs(model, sequences, encoder_input=None):
    if encoder_input is None:
        model_kwargs = {}
    else:
        encoder_input, model_input_name, model_kwargs = model._prepare_model_inputs(
            encoder_input.input_ids, model.config.bos_token_id, {'attention_mask': encoder_input.attention_mask},
        )
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            encoder_input, model_kwargs, model_input_name
        )

    device = sequences.device
    if len(sequences.shape) == 1:
        sequences = sequences.unsqueeze(0)
    batch_size, seq_len = sequences.shape
    logp = torch.zeros(batch_size, device=device)
    transition_scores = torch.zeros(batch_size, seq_len - 1, device=device)
    finished = torch.zeros(batch_size, device=device, dtype=torch.bool)
    with (torch.no_grad()):
        decoder_input_ids = _prepare_decoder_input(model, sequences)
        # decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], dtype=torch.long, device=device)
        # assert torch.all(decoder_input_ids == alt_decoder_input_ids)
        for i in range(seq_len - 1):
            outputs = model(**model_kwargs, decoder_input_ids=decoder_input_ids)
            last_logits = outputs.logits[:, -1, :]
            last_logits = last_logits.log_softmax(dim=-1)
            next_token = sequences[:, i+1].view(batch_size,)
            logp += last_logits.gather(dim=-1, index=next_token.unsqueeze(-1)).squeeze(-1) * (~finished).float()
            transition_scores[:, i] = last_logits.gather(dim=-1, index=next_token.unsqueeze(-1)).squeeze(-1) * (~finished).float()
            finished = finished | (next_token == model.config.eos_token_id)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(-1)], dim=1)
    return logp, transition_scores


def compute_logprobs_from_scores(model, outputs):
    logp_computed = model.compute_transition_scores(sequences=outputs.sequences, scores=outputs.scores, normalize_logits=True)
    nonpad_mask = outputs.sequences[:, 1:] != model.config.pad_token_id
    result = torch.where(nonpad_mask, logp_computed, torch.zeros_like(logp_computed))
    return result


def test_t5():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    inputs = tokenizer(["translate English to German: What is your name, my dear Friend? I missed you so much",
                           "translate English to German: How old are you?",
                          ],
                          return_tensors="pt", padding=True, truncation=True
                          )
    outputs = model.generate(**inputs, num_beams=1, do_sample=True, num_return_sequences=1,
                             return_dict_in_generate=True, output_scores=True,
                             top_k=0,
                             # length_penalty=0.0, early_stopping=True,
                             max_new_tokens=100)
    # gen_logp = outputs.sequences_scores
    print(f"generated text: {tokenizer.batch_decode(outputs.sequences)}")
    print(f"generated tokens: {outputs.sequences}")
    # print(f"generated logp: {gen_logp}")
    _, gold_transitions = compute_true_logprobs(model, outputs.sequences, encoder_input=inputs)
    # logp_computed = model.compute_transition_scores(sequences=outputs.sequences, scores=outputs.scores,
    #                                                 beam_indices=outputs.beam_indices,
    #                                                 normalize_logits=True)
    logp_computed = compute_logprobs_from_scores(model, outputs)
    print(f"score transitions: {logp_computed.sum(dim=1)}")
    print(f"gold transitions:  {gold_transitions.sum(dim=1)}")



if __name__ == "__main__":
    test_t5()
