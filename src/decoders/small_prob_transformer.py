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
        self.vocab_uncomplete_seqs[self.config.eos_token_id] = -float("inf")
        self.vocab_finished_seqs = torch.tensor(-float("inf"), device=device).repeat(self.config.vocab_size)
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
            mask_inf_uncomplete = uncomplete_seqs.unsqueeze(1) * self.vocab_uncomplete_seqs
            mask_inf_finished = uncomplete_seqs.logical_not().unsqueeze(1) * self.vocab_finished_seqs
            result = torch.where(uncomplete_seqs.unsqueeze(1), mask_inf_uncomplete, mask_inf_finished)
            output[:, seq_len, :] = result
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
