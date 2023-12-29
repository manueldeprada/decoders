from typing import List, Union, Optional

import torch
from transformers import StoppingCriteriaList

from .beam_search import BeamSearchDecoder
from .simple_beam_utils import BeamSearchNode
from ..strategies.sbs_helpers.gumbel import gumbel, gumbel_with_maximum
from ..strategies.sbs_helpers.logits_process import LogitsProcessor, LogitsProcessorList


class SimpleSBSLogitProcessor(LogitsProcessor):

    def __call__(self, input_ids, scores, **kwargs):
        """
        LogitsProcessor that implements stochastic beam search.

        :param input_ids: shape (batch_size * num_beams, seq_len)
        :param scores: shape (batch_size * num_beams, vocab_size)
        :param kwargs.beam_log_probs: shape (batch_size * num_beams,)
        :param kwargs.past_processed_scores: tuple of size seq_len with tensors (batch_size * num_beams, vocab_size)
        :return: shape (batch_size, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if kwargs["nodes"] is None:
            # no nodes -> we are in the first step of generation. input_ids is of shape (batch_size, 1)
            last_gumbels = gumbel(size=(batch_size,)).to(input_ids.device)  # shape (batch_size, )
            log_probs = scores
        else:
            # input_ids is of shape (alive_beams,seq_len)
            nodes: List[BeamSearchNode] = kwargs['nodes']  # list of size num_active_beams
            last_gumbels = torch.tensor([node.last_score for node in nodes], device=device)
            beam_log_probs = torch.tensor([node.log_prob for node in nodes], device=device)
            log_probs = scores + beam_log_probs.unsqueeze(-1)  # shape (num_active_beams, vocab_size)

        new_gumbels, _ = gumbel_with_maximum(log_probs, last_gumbels)  # shape (num_active_beams, vocab_size)
        return new_gumbels


class SimpleStochasticBeamSearchDecoder(BeamSearchDecoder):

    def __call__(self,
                 model: Union["PreTrainedModel", "GenerationMixin"],
                 input_ids: torch.LongTensor,
                 logits_processor: Optional[LogitsProcessorList] = None,
                 stopping_criteria: Optional[StoppingCriteriaList] = None,
                 keep_k_always_alive: Optional[int] = False,
                 eval_by_score: Optional[bool] = False,
                 **model_kwargs,
                 ):
        r"""
        Simple wrapper around the Beam Search decoder that appends the stochastic beam search logits processor.
        """
        if logits_processor is not None:
            logits_processor = LogitsProcessorList(logits_processor)
            if any(isinstance(p, SimpleSBSLogitProcessor) for p in logits_processor):
                pass
            else:
                logits_processor.append(SimpleSBSLogitProcessor())
        else:
            logits_processor = LogitsProcessorList([SimpleSBSLogitProcessor()])
        return super().__call__(model, input_ids, logits_processor, stopping_criteria, keep_k_always_alive,
                                eval_by_score, **model_kwargs)
