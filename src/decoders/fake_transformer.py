from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizer, BatchEncoding
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch


class FakeTransformerConfig(PretrainedConfig):
    model_type = "FakeTransformer"

    def __init__(self, vocab_size=4, **kwargs):
        super().__init__(pad_token_id=-1, eos_token_id=3, bos_token_id=0, **kwargs)
        self.vocab_size = vocab_size
        self.max_new_tokens = 4
        self.do_sample = False
        self.num_beams = 1


class FakeTransformer(PreTrainedModel):
    config_class = FakeTransformerConfig

    def __init__(self, config=None):
        config = FakeTransformerConfig() if config is None else config
        super().__init__(config)
        self.fake_param = torch.nn.Parameter(torch.tensor(1.0))  # need at least one parameter to be a valid model

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
        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        # Placeholder for output probabilities
        output = torch.zeros(batch_size, seq_len, self.config.vocab_size)

        if seq_len >= 1:
            output[:, 0, 1] = 0.75
            output[:, 0, 2] = 0.25

        if seq_len >= 2:
            output[:, 1, 1] = 0.10
            output[:, 1, 2] = 0.90

        if seq_len >= 3:
            output[:, 2, :] = 0.0
            output[:, 2, 3] = 1.0

        if not past_key_values:  # when using past_key_values, only last token logits are generated
            output = output[:, -1, :].unsqueeze(1)

        # Placeholder for past_key_values, attentions, hidden_states
        past_key_values = () if past_key_values is None else past_key_values
        attentions = () if output_attentions else None
        hidden_states = () if output_hidden_states else None

        output = output.log()
        #replace -inf with -1e9
        #output[output == float('-inf')] = -1e9

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


class FakeTokenizer(PreTrainedTokenizer):
    def __init__(self, bos_token=0, num_seq=2, **kwargs):
        super().__init__(**kwargs)
        self.bos_token = bos_token
        self.num_seq = num_seq

    def __call__(self, text, **kwargs):
        tensor = torch.tensor([[self.bos_token]] * self.num_seq) #  default torch.tensor([[0],[0]])
        return BatchEncoding({"input_ids": tensor})

    def batch_decode(self, token_ids, **kwargs):
        return [str(ids) for ids in token_ids]


if __name__ == "__main__":
    config = FakeTransformerConfig()
    model = FakeTransformer(config)

    input_ids = torch.tensor([[0]] * 10)
    output = model.generate(input_ids, max_length=4, do_sample=True, return_dict_in_generate=True, output_scores=True)
    print(f"generated tokens: {output.sequences}")
    print(f"probs at step 1: {torch.exp(output.scores[0][0, :])}")
    print(f"probs at step 2: {torch.exp(output.scores[1][0, :])}")
    print(f"probs at step 3: {torch.exp(output.scores[2][0, :])}")
