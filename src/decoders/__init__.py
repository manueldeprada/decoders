from .binary_transformer import BinaryCodeTransformer, BinaryCodeTransformerConfig, batched_bin_vec_to_decimal, decimal_to_bin_vec
from .fake_transformer import FakeTransformer
from .generation_mixin import GenerationMixin
from .small_prob_transformer import SmallProbTransformer, SmallProbTransformerConfig
# from .strategies.stochastic_beam_search import OldStochasticBeamSearchDecoder
from .toolbox import compute_true_logprobs
from .simple.ancestral import SamplingDecoder
from .simple.beam_search import BeamSearchDecoder
from .simple.stochastic_beam_search import StochasticBeamSearchDecoder
from .simple.stochastic_beam_search import SimpleSBSLogitProcessor
from .strategies.sbs_helpers.logits_process import LogitsProcessorList

def inject_supervitamined_decoders(model: "PreTrainedModel"):
    import types
    model.generate = types.MethodType(GenerationMixin.generate, model)
    # model._contrastive_search = types.MethodType(GenerationMixin._contrastive_search, model)
    # model._greedy_search = types.MethodType(GenerationMixin.greedy_search, model)
    # model._sample = types.MethodType(GenerationMixin.sample, model)
    # model._beam_search = types.MethodType(GenerationMixin.beam_search, model)
    # model._beam_sample = types.MethodType(GenerationMixin.beam_sample, model)
    # model._group_beam_search = types.MethodType(GenerationMixin.group_beam_search, model)
    # model._constrained_beam_search = types.MethodType(GenerationMixin.constrained_beam_search, model)
    # model._assisted_decoding = types.MethodType(GenerationMixin.assisted_decoding, model)
