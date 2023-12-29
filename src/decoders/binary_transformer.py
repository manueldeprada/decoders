import numpy as np
import torch
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput


class BinaryCodeTransformerConfig(PretrainedConfig):
    """Configuration class with appropriate defaults for BinaryCodeTransformer."""
    model_type = "BinaryCodeTransformer"

    def __init__(self, max_new_tokens=100, vocab_size=3, **kwargs):
        super().__init__(pad_token_id=-5, eos_token_id=2, bos_token_id=-1, **kwargs)
        self.vocab_size = vocab_size
        self.max_new_tokens = max_new_tokens
        self.do_sample = True
        self.num_beams = 1


# Helper functions

@torch.jit.script
def batched_bin_vec_to_decimal(vec):
    """Convert rows with binary digits to decimal natural numbers.
    Example: tensor([[0, 1, 0, 1, 1, 0]) -> tensor([22])"""
    if torch.all(vec[..., -1] == 2):
        vec = vec[..., :-1]
    return torch.matmul(vec.long(), (2 ** torch.arange(vec.size(-1) - 1, -1, -1))).long()


def decimal_to_bin_vec(num, n):
    """Convert decimal natural number to binary vector."""
    tensor_without_eos = torch.tensor(list(map(int, bin(num)[2:].zfill(n))))
    return torch.cat((tensor_without_eos, torch.tensor([2])))


def get_binary_completions(seq, max_len):
    """Get all binary completions of a sequence up to a given length.
    Example: seq=[0, 1], max_len=4 -> [[0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1]]"""
    seq_len = seq.shape[-1]
    empty_bits = max_len - seq_len
    seqs = torch.zeros(2 ** empty_bits, max_len)
    seqs[:, :seq_len] = seq.repeat(1, 2 ** empty_bits, 1)
    seqs[:, seq_len:] = torch.tensor([list(map(int, bin(i)[2:].zfill(empty_bits))) for i in range(2 ** empty_bits)])
    return seqs


def hash_seq(seq):
    """Hash a sequence of binary digits."""
    return hash(tuple(seq.reshape(-1).tolist()))
    # return batched_bin_vec_to_decimal(seq).item()*1000 + seq.shape[-1]


class BinaryCodeTransformer(PreTrainedModel):
    """A model stub that samples binary sequences
    of a fixed length n with a predetermined discrete
    distribution over all 2^n possible sequences.

    The model maps each sequence to its decimal
    representation as an integer from 0 to 2^n-1"""
    config_class = BinaryCodeTransformerConfig

    def __init__(self, config=None, n=6, prob_vector=None):
        """Initialize the model.

        :param config: BinaryCodeTransformerConfig
        :param n: length of binary sequences (there will be 2^n possible sequences)
        :param prob_vector: a vector of probabilities over all possible sequences,
                            indexed by their decimal representation
        """
        config = BinaryCodeTransformerConfig() if config is None else config
        super().__init__(config)
        self.fake_param = torch.nn.Parameter(torch.tensor(1.0))  # need at least one parameter to be a valid model
        self.n = n
        self.num_seqs = 2 ** self.n
        if prob_vector is not None:
            assert len(prob_vector) == self.num_seqs
            assert torch.allclose(torch.tensor(prob_vector).sum(), torch.tensor(1.0))
            self.p = torch.tensor(prob_vector, dtype=torch.float32)
        else:
            self.p = torch.tensor(np.random.dirichlet(np.ones(self.num_seqs)), dtype=torch.float32)
        self.prob_mem = {}
        self.precompute_probs()

    def precompute_probs(self):
        """Precompute probabilities of all possible sequences"""
        seq = torch.tensor([[-1]])
        seqs = [s for i in range(2, self.n + 2) for s in get_binary_completions(seq, i)]
        probs = [self.seq_prob(seq[1:]) for seq in seqs]
        for seq, prob in zip(seqs, probs):
            self.prob_mem[hash_seq(seq[1:])] = prob

    def seq_prob(self, seq):
        """Compute the probability of a binary sequence.
        When it is incomplete, sum over all possible completions."""
        assert len(seq.shape) == 1
        seq_len = seq.shape[-1]
        if seq_len == self.n:
            value = self.p[batched_bin_vec_to_decimal(seq)]
        else:
            seqs = get_binary_completions(seq, self.n)
            values = batched_bin_vec_to_decimal(seqs)
            value = self.p[values].sum()
        return value

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            return_dict=True,
            output_attentions=None,
            output_hidden_states=None,
            **kwargs,
    ):
        assert past_key_values is None
        batch_size, seq_len = input_ids.shape

        # append an extra middle dim for the vocab size
        def next_seqs(seqs):
            expanded_t = seqs.unsqueeze(1).expand(-1, 2, -1)
            new_dim = torch.zeros(batch_size, 2, 1)
            new_dim[:, 0, :] = 0
            new_dim[:, 1, :] = 1
            return torch.cat((expanded_t, new_dim), dim=2)

        if input_ids.shape[1] == self.n + 1:
            output = torch.zeros(batch_size, 1, 3)
            output[:, 0, 2] = 1.0
        else:
            continuations = next_seqs(input_ids).view(batch_size * 2, seq_len + 1)
            continuations = continuations[:, 1:].unbind()
            continuations_probs = torch.tensor([self.prob_mem[hash_seq(seq)] for seq in continuations],
                                               dtype=torch.float32)
            # output[:, seq_len - 1, :] = continuations_probs.view(batch_size, self.config.vocab_size)
            output = continuations_probs.view(batch_size, 2).unsqueeze(1)
            output = torch.cat([output, torch.zeros(batch_size, 1, 1)], dim=-1)
        # Placeholder for past_key_values, attentions, hidden_states
        past_key_values = () if past_key_values is None else past_key_values
        attentions = () if output_attentions else None
        hidden_states = () if output_hidden_states else None

        output = output.log()
        # replace -inf with -1e9
        # output[output == float('-inf')] = -1e9

        # Create Seq2SeqLMOutput object
        if not return_dict:
            att, hidd = (
                attentions if attentions is not None else (),
                hidden_states if hidden_states is not None else (),
            )
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


def demo():
    config = BinaryCodeTransformerConfig(max_new_tokens=6)
    model = BinaryCodeTransformer(config)

    input_ids = torch.tensor([[-1]] * 10)
    output = model.generate(
        input_ids,
        return_dict_in_generate=True,
        output_scores=True,
    )
    print(f"generated tokens: {output.sequences}")
    print(f"probs of seq0 at step 1: {torch.exp(output.scores[0][0, :])}")
    print(f"probs of seq0 at step 2: {torch.exp(output.scores[1][0, :])}")
    print(f"probs of seq0 at step 3: {torch.exp(output.scores[2][0, :])}")
    print(f"probs of seq0 at step 4: {torch.exp(output.scores[3][0, :])}")
    print(f"probs of seq0 at step 5: {torch.exp(output.scores[4][0, :])}")
    print(f"probs of seq0 at step 6: {torch.exp(output.scores[5][0, :])}")
    print(f"probs predetermined seq0: {model.p[batched_bin_vec_to_decimal(output.sequences[0])]}")


def test():
    torch.manual_seed(0)
    np.random.seed(0)
    config = BinaryCodeTransformerConfig(max_new_tokens=6)
    model = BinaryCodeTransformer(config)

    input_ids = torch.tensor([[-1]] * 10)
    output = model.generate(input_ids, return_dict_in_generate=True, output_scores=True)
    torch.set_printoptions(precision=8)
    assert torch.allclose(
        model.p,
        torch.tensor(
            [0.01277674, 0.02016235, 0.01482116, 0.01263750, 0.00884638, 0.01666631, 0.00923922, 0.03569583,
                0.05321665, 0.00776361, 0.02518661, 0.01208322, 0.01347602, 0.04171163, 0.00118292, 0.00146347,
                0.00032791, 0.02869582, 0.02417343, 0.03275462, 0.06173009, 0.02577006, 0.00993612, 0.02434604,
                0.00202075, 0.01639778, 0.00248399, 0.04646620, 0.01184487, 0.00859782, 0.00493299, 0.02389202,
                0.00977804, 0.01349049, 0.00030452, 0.01543374, 0.01520281, 0.01540431, 0.04620123, 0.01838373,
                0.00715223, 0.00922338, 0.01920197, 0.00099718, 0.01764163, 0.01782921, 0.00379200, 0.00221587,
                0.00608375, 0.00725792, 0.01355620, 0.00926820, 0.07151123, 0.00172794, 0.00376141, 0.00282407,
                0.01699675, 0.00468898, 0.01008079, 0.00449949, 0.00277934, 0.00187757, 0.01714653, 0.00238738,
            ], dtype=torch.float32),
    )
    print(f"scores: {output.scores[5]}")
    assert torch.allclose(
        output.scores[5],
        torch.tensor(
            [
                [-5.58296156, -5.86957741],  # each column is a different sample
                [-2.78498363, -3.65854216],
                [-2.63790083, -6.36082315],
                [-4.61157894, -3.71538639],
                [-4.18627453, -4.17310810],
                [-4.03749371, -4.02691698],
                [-4.43586063, -4.75624657],
                [-3.68144274, -4.41593790],
                [-4.30684376, -3.17697525],
                [-8.02278233, -3.55100393],
            ],
            dtype=torch.float32,
        ),
    )
    print("test passed")


def benchmark():
    import timeit
    from .fake_transformer import FakeTransformerConfig
    from .fake_transformer import FakeTransformer

    models = {
        "BinaryCodeTransformer": BinaryCodeTransformer(BinaryCodeTransformerConfig(max_new_tokens=6)),
        "FakeTransformer": FakeTransformer(FakeTransformerConfig()),
    }
    inputs = {
        "BinaryCodeTransformer": torch.tensor([[-1]] * 1000),
        "FakeTransformer": torch.tensor([[0]] * 1000),
    }

    def run(model, inputs):
        model.generate(
            inputs,
            return_dict_in_generate=True,
            output_scores=True,
        )

    for name, model in models.items():
        print(f"{name}: {timeit.timeit(lambda: run(model, inputs[name]), number=10)}")


if __name__ == "__main__":
    import typer

    app = typer.Typer()
    app.command()(demo)
    app.command()(benchmark)
    app.command()(test)
    app()
