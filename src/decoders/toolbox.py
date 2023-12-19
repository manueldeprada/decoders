import torch


def _prepare_decoder_input(model, sequences):
    batch_size = sequences.shape[0]
    device = sequences.device
    start_token = model.config.decoder_start_token_id
    assert torch.all(sequences[:, 0] == start_token)
    decoder_input = torch.tensor([[start_token]], dtype=torch.long, device=device).repeat(batch_size, 1)
    return decoder_input.view(batch_size, 1)


def compute_true_logprobs(model, sequences, encoder_input=None):
    encoder_input = {} if encoder_input is None else encoder_input
    batch_size = sequences.shape[0]
    device = sequences.device
    if len(sequences.shape) == 1:
        sequences = sequences.unsqueeze(0)
    logp = torch.zeros(batch_size, device=sequences.device, dtype=torch.float64)
    finished = torch.zeros(batch_size, device=sequences.device, dtype=torch.bool)
    with (torch.no_grad()):
        decoder_input_ids = _prepare_decoder_input(model, sequences)
        # decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], dtype=torch.long, device=device)
        # assert torch.all(decoder_input_ids == alt_decoder_input_ids)
        for i in range(sequences.shape[1]-1):
            outputs = model(**encoder_input, decoder_input_ids=decoder_input_ids)
            last_logits = outputs.logits[:, -1, :]
            last_logits = last_logits.log_softmax(dim=-1)
            next_token = sequences[:, i+1].view(batch_size,).cpu()
            logp += last_logits.gather(dim=-1, index=next_token.unsqueeze(-1)).squeeze(-1) * (~finished).float()
            finished = finished | (next_token == model.config.eos_token_id)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(-1)], dim=1)
    return logp


def test_t5():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    inputs = tokenizer(["translate English to German: What is your name, my dear Friend? I missed you so much",
                           "translate English to German: How old are you?",
                          ],
                          return_tensors="pt", padding=True, truncation=True
                          )
    outputs = model.generate(**inputs, num_beams=2, do_sample=False, num_return_sequences=1,
                             return_dict_in_generate=True, output_scores=True,
                             length_penalty=0.0, early_stopping=True,
                             max_new_tokens=100)
    gen_logp = outputs.sequences_scores
    print(f"generated text: {tokenizer.batch_decode(outputs.sequences)}")
    print(f"generated tokens: {outputs.sequences}")
    print(f"generated logp: {gen_logp}")
    logp = compute_true_logprobs(model, outputs.sequences, encoder_input=inputs)
    logp_computed = model.compute_transition_scores(sequences=outputs.sequences, scores=outputs.scores,
                                                    beam_indices=outputs.beam_indices,
                                                    normalize_logits=True)
    print(f"compute_transition logp: {logp_computed.sum(dim=1)}")
    print(f"true logp: {logp}")



if __name__ == "__main__":
    test_t5()
