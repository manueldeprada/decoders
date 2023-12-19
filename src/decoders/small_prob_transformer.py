from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizer, BatchEncoding
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch


class SmallProbTransformerConfig(PretrainedConfig):
    model_type = "SmallProbTransformer"

    def __init__(self, **kwargs):
        super().__init__(pad_token_id=10, eos_token_id=10, bos_token_id=-1, **kwargs)
        # vocab: 0-9. token 10 is pad and eos
        self.vocab_size = 11
        self.real_vocab_size = 10
        self.max_new_tokens = 150
        self.do_sample = False
        self.num_beams = 1
        self.decoder_start_token_id = -2


def route2length(route):
    """Converts a route to a length.
    each seq of a route i has prob 10^(-13*(i+1)).
    Hence, there is 10^(13*(i+1)-1) sequences of length i (as they sum to 0.1 probability).
    """
    return ((route + 1) * 13) - 1

def f_experiment(seq):
    """Computes the function 0 -> 100000, 1 -> 10000, 2 -> 1000, 3 -> 100, 4 -> 10, 5 -> 1
    6 -> 0.1, 7 -> 0.01, 8 -> 0.001, 9 -> 0.0001
    """
    return 10 ** (5 - seq)


class SmallProbTransformer(PreTrainedModel):
    config_class = SmallProbTransformerConfig

    def __init__(self, config=None):
        config = SmallProbTransformerConfig() if config is None else config
        super().__init__(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.SmallProb_param = torch.nn.Parameter(torch.tensor(1.0))  # need at least one parameter to be a valid model
        self.vocab_uncomplete_seqs = torch.tensor(1.0 / self.config.real_vocab_size,
                                                  device=device).repeat(self.config.vocab_size).log()
        self.vocab_uncomplete_seqs[self.config.eos_token_id] = -1e9
        self.vocab_finished_seqs = torch.tensor(-1e9, device=device).repeat(self.config.vocab_size)
        self.vocab_finished_seqs[self.config.eos_token_id] = 0.0

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            return_dict=True,
            output_attentions=None,
            output_hidden_states=None,
            **kwargs
    ):
        if input_ids is None:
            assert kwargs["decoder_input_ids"] is not None, "You have to specify either decoder_input_ids or input_ids"
            input_ids = kwargs["decoder_input_ids"]
        seq_len = input_ids.shape[1] - 1
        batch_size = input_ids.shape[0]

        # Placeholder for output probabilities
        output = torch.zeros(batch_size, seq_len + 1, self.config.vocab_size, device=input_ids.device)

        # print(f"seq_len: {seq_len}")

        if seq_len > 0:
            routes = input_ids[:, 1]
            target_lens = route2length(routes)
            # print(f"routes: {routes}")
            # print(f"target_lens: {target_lens}")
            uncomplete_seqs = seq_len < target_lens
            # print(f"uncomplete_seqs: {uncomplete_seqs.int()}")
            finished_seqs = uncomplete_seqs.logical_not()
            output[:, seq_len, :] = uncomplete_seqs.unsqueeze(1) * self.vocab_uncomplete_seqs \
                                        + finished_seqs.unsqueeze(1) * self.vocab_finished_seqs
        else:
            output[:, 0, :] = self.vocab_uncomplete_seqs

        if not past_key_values:  # when using past_key_values, only last token logits are generated
            output = output[:, -1, :].unsqueeze(1)

        # Placeholder for past_key_values, attentions, hidden_states
        past_key_values = () if past_key_values is None else past_key_values
        attentions = () if output_attentions else None
        hidden_states = () if output_hidden_states else None

        # output = output.log()
        # replace -inf with -1e9
        # output[output == float('-inf')] = -1e9

        # Create Seq2SeqLMOutput object
        if not return_dict:
            att, hidd = attentions if attentions is not None else (), hidden_states if hidden_states is not None else ()
            return (output,) + past_key_values + hidd + att
        else:
            return Seq2SeqLMOutput(
                loss=None,
                logits=output,
                past_key_values=past_key_values,
                decoder_hidden_states=hidden_states,
                decoder_attentions=attentions,
                encoder_last_hidden_state=None,
                encoder_hidden_states=None,
                encoder_attentions=None,
            )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def _reorder_cache(self, past, beam_idx):
        return past


class SmallProbTokenizer(PreTrainedTokenizer):
    def __init__(self, bos_token=0, num_seq=2, **kwargs):
        super().__init__(**kwargs)
        self.bos_token = bos_token
        self.num_seq = num_seq

    def __call__(self, text, **kwargs):
        tensor = torch.tensor([[self.bos_token]] * self.num_seq)  # default torch.tensor([[0],[0]])
        return BatchEncoding({"input_ids": tensor})

    def batch_decode(self, token_ids, **kwargs):
        return [str(ids) for ids in token_ids]


if __name__ == "__main__":
    from decoders.strategies.sbs_helpers.logits_process import LogitsProcessorList, TemperatureLogitsWarper
    from decoders import inject_supervitamined_decoders, StochasticBeamSearchDecoder, toolbox
    config = SmallProbTransformerConfig()
    model = SmallProbTransformer(config)
    inject_supervitamined_decoders(model)

    input_ids = torch.tensor([[-2]])  # for sampling and on cpu, this is faster than num_return_sequences
    output = model.generate(input_ids,
                            generation_strategy=StochasticBeamSearchDecoder(), num_beams=100, num_return_sequences=100,
                            length_penalty=0.0,
                            early_stopping=True,
                            #logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature=0.0001)]),
                            #num_beams=1,
                            do_sample=False, return_dict_in_generate=True,
                            output_scores=True)
    # print(f"generated tokens: {output.sequences}")
    # print(f"routes: {output.sequences[:, 1]}")
    routes_freq = output.sequences[:, 1].bincount(minlength=config.real_vocab_size)
    print(f"routes freq: {routes_freq}, total: {routes_freq.sum()}")
    r_lens = route2length(output.sequences[:, 1])

    for i in range(0,10):
        print(f"Route {i}, len {route2length(i)}, freq {routes_freq[i]}")

        route_i_seqs = output.sequences[:, 1] == i
        if i==0:
            print(output.sequences[route_i_seqs, :][0:10])
        computed_probs = model.compute_transition_scores(sequences=output.sequences, scores=output.scores,
                                                         normalize_logits=True, beam_indices=output.beam_indices)
        gen_probs = output.sequences_scores
        print(f"probs (generate): {gen_probs[route_i_seqs].mean()}")
        print(f"probs (computed): {computed_probs[route_i_seqs, :].sum(dim=1).mean()}")
        own_probs = toolbox.compute_true_logprobs(model, output.sequences)
        print(f"probs (own): {own_probs[route_i_seqs].mean()}")


    # print(f"probs at step 1: {torch.exp(output.scores[0][0, :])}")
    # print(f"probs at step 2: {torch.exp(output.scores[1][0, :])}")
    # print(f"probs at step 3: {torch.exp(output.scores[2][0, :])}")

    #lets estimate f with monte carlo
    true_f = f_experiment(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).double()).sum()/10
    print(f"true f: {true_f}")
    f_mc = f_experiment(output.sequences[:, 1].double()).mean()
    print(f"estimated f: {f_mc}")
    error = (f_mc - true_f).abs()
    print(f"error: {error}, relative error: {error / true_f}")

