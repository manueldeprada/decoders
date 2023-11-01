import types

from transformers import PreTrainedModel

from .utils import GenerationMixin
from .strategies.stochastic_beam_search import StochasticBeamSearchDecoder
from .fake_transformer import FakeTransformer
from .binary_transformer import BinaryCodeTransformer, batched_bin_vec_to_decimal, decimal_to_bin_vec


def inject_supervitamined_decoders(model: PreTrainedModel):
    model.generate = types.MethodType(GenerationMixin.generate, model)
    model.contrastive_search = types.MethodType(GenerationMixin.contrastive_search, model)
    model.greedy_search = types.MethodType(GenerationMixin.greedy_search, model)
    model.sample = types.MethodType(GenerationMixin.sample, model)
    model.beam_search = types.MethodType(GenerationMixin.beam_search, model)
    model.beam_sample = types.MethodType(GenerationMixin.beam_sample, model)
    model.group_beam_search = types.MethodType(GenerationMixin.group_beam_search, model)
    model.constrained_beam_search = types.MethodType(GenerationMixin.constrained_beam_search, model)
    model.assisted_decoding = types.MethodType(GenerationMixin.assisted_decoding, model)
