from typing import Optional, Union, TYPE_CHECKING
import torch
from transformers import GenerationConfig, StoppingCriteriaList
from .simple_beam_utils import SimpleBeamSearch, BeamSearchNode, pad_tensors
from ..strategies.sbs_helpers.logits_process import LogitsProcessorList
from ..strategies.utils import GenerationStrategy, GenerateBeamDecoderOnlyOutput
from ..strategies.beam_decoder import _split_model_inputs, stack_model_outputs

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel, GenerationMixin


class BeamSearchDecoder(GenerationStrategy):

    def __init__(self, config: GenerationConfig = None, check_config: bool = True):
        super().__init__(config, check_config=check_config)

    @classmethod
    def param_check(cls, config: GenerationConfig):
        pass

    def __call__(self,
                 model: Union["PreTrainedModel", "GenerationMixin"],
                 input_ids: torch.LongTensor,
                 logits_processor: Optional[LogitsProcessorList] = None,
                 stopping_criteria: Optional[StoppingCriteriaList] = None,
                 synced_gpus: bool = False,
                 streamer: Optional["BaseStreamer"] = None,
                 keep_k_always_alive: Optional[int] = False,
                 disable_kv_cache: Optional[bool] = False,
                 eval_by_score: Optional[bool] = False,
                 encoder_input_ids: Optional[torch.LongTensor] = None,
                 gen_args: dict = None,
                 **model_kwargs,
                 ):
        r"""
        Simple implementation of Beam Search.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            keep_k_always_alive (`int`, *optional*, defaults to `False`):
                If set to `True`, always keep at least `num_beams` hypotheses alive, irrespective of number
                of finished hypotheses.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.SampleDecoderOnlyOutput`] or [`~generation.SampleEncoderDecoderOutput`].
        """
        from .stochastic_beam_search import SimpleSBSLogitProcessor
        if logits_processor is not None and any(isinstance(p, SimpleSBSLogitProcessor) for p in logits_processor) and not eval_by_score:
            print("WARNING: SimpleSBSLogitProcessor is used but eval_by_score is False.")

        logits_processor = LogitsProcessorList(logits_processor) if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        # init values
        pad_token_id = self.config._pad_token_tensor
        batch_size, _ = input_ids.shape
        num_beams = self.config.num_beams
        vocab_size = model.config.vocab_size if "vocab_size" in model.config.__dict__ else model.config.decoder.vocab_size
        
        # 0. Initialize beam searches, one for each sequence in the batch
        searches = [SimpleBeamSearch(num_beams, model.config.eos_token_id, eval_by_score) for _ in range(batch_size)]

        # 1. Generate initial hypotheses
        model_kwargs = model._get_initial_cache_position(input_ids, model_kwargs)
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if encoder_input_ids is not None:
            model_inputs["input_ids"] = encoder_input_ids
        outputs = model(**model_inputs, return_dict=True)
        model_kwargs = model._update_model_kwargs_for_generation(outputs, model_kwargs,
                                                                 is_encoder_decoder=model.config.is_encoder_decoder)
        model_states = _split_model_inputs(model_kwargs, split_size=1, full_batch_size=batch_size, config=model.config.get_text_config())
            
            
        logits = outputs.logits[:, -1, :]
        
        if self.__class__.__name__ == "StochasticBeamSearchDecoder":
            if len(logits_processor) > 1:
                prev_logit_processors = logits_processor[:-1]
                sbs_processor = logits_processor[-1]                    
                logits = LogitsProcessorList(prev_logit_processors)(input_ids, logits, nodes=None).log_softmax(dim=-1)
                scores = sbs_processor(input_ids, logits, nodes=None)
            else:
                logits = logits.log_softmax(dim=-1)
                scores = logits_processor(input_ids, logits, nodes=None)
        else:
            logits = logits.log_softmax(dim=-1)
            scores = logits_processor(input_ids, logits, nodes=None)
                    
        scores, next_candidates = torch.topk(scores, min(num_beams, vocab_size))  # (batch_size, num_beams)
        log_probs = torch.gather(logits, -1, next_candidates)  # order by scores, store true log probs

        # 1.1. Add initial hypotheses to the beam searches
        for b_idx in range(batch_size):
            for j in range(min(num_beams, vocab_size)):
                next_token = next_candidates[b_idx, j]
                new_seq = torch.cat((input_ids[b_idx], next_token.unsqueeze(0)))
                node = BeamSearchNode(searches[b_idx], new_seq, model_states[b_idx], log_probs[b_idx, j],
                                      scores[b_idx, j])
                searches[b_idx].add(node, finished=(next_token == model.config.eos_token_id))

        # 2. Beam search generation loop
        while True:
            # 2.1 Pop the current nodes to expand
            nodes = [n for s in searches for n in s.get_current_beams()]
            if len(nodes) == 0:
                break  # All beams ended in EOS

            # 2.2 Expand nodes, get top `num_beams` candidates
            input_ids = pad_tensors([node.sequence for node in nodes], pad_token_id)
            input_ids = torch.stack(input_ids, dim=0)
            model_args = stack_model_outputs([node.model_state for node in nodes], config=model.config.get_text_config())
            model_inputs = model.prepare_inputs_for_generation(input_ids, **model_args)
            if encoder_input_ids is not None:
                model_inputs["input_ids"] = encoder_input_ids
            outputs = model(**model_inputs, return_dict=True)
            
            model_args = model._update_model_kwargs_for_generation(outputs, model_args,
                                                                 is_encoder_decoder=model.config.is_encoder_decoder)
            model_args = _split_model_inputs(model_args, split_size=1, full_batch_size=len(nodes), config=model.config.get_text_config())

            logits = outputs.logits[:, -1, :]
            if self.__class__.__name__ == "StochasticBeamSearchDecoder":
                if len(logits_processor) > 1:
                    prev_logit_processors = logits_processor[:-1]
                    sbs_processor = logits_processor[-1]                    
                    logits = LogitsProcessorList(prev_logit_processors)(input_ids, logits, nodes=nodes).log_softmax(dim=-1)
                    scores = sbs_processor(input_ids, logits, nodes=nodes)
                else:
                    logits = logits.log_softmax(dim=-1)
                    scores = logits_processor(input_ids, logits, nodes=nodes)
            else:
                logits = logits.log_softmax(dim=-1)
                scores = logits_processor(input_ids, logits, nodes=nodes)
            scores, next_candidates = torch.topk(scores, min(num_beams, vocab_size))  # (batch_size, num_beams)
            log_probs = torch.gather(logits, -1, next_candidates)  # order by scores, store true log probs

            # 2.3 Push new candidates to the beam searches
            for n_idx, node in enumerate(nodes):
                # model_state = update_model_kv_cache(node.model_state, n_idx, model_args, disable_kv_cache)
                for j in range(min(num_beams, vocab_size)):
                    next_token = next_candidates[n_idx, j]
                    beam_log_p = node.log_prob + log_probs[n_idx, j]
                    new_seq = torch.cat((input_ids[n_idx], next_token.unsqueeze(0)))
                    new_node = BeamSearchNode(node.search, new_seq, model_args[n_idx], beam_log_p, scores[n_idx, j])
                    new_node.search.add(new_node, finished=(next_token == model.config.eos_token_id))

            # 2.4 Prune and check stopping criteria
            for search in searches:
                search.prune(keep_k_always_alive=keep_k_always_alive)

            if all(stopping_criteria(input_ids, scores)):
                break

            # rationale: if we always keep k non-eos beams alive, generation only stops by max_length
            # instead, with this heuristic, we mimic HF's early stopping: when we have k finished hypotheses, we allow
            # one more round of expansion to allow for EOS to be generated in alive beams.
            # THIS A HEURISTIC to test against HF's implementation. keep_k_always_alive should be False for correctness.
            if keep_k_always_alive and all([search.is_done() for search in searches]):
                keep_k_always_alive = False
        # 3. Return best hypotheses
        best_sents = []
        best_log_probs = []
        last_scores = []
        for search in searches:
            for node in search.final_best_n(self.config.num_return_sequences):
                best_sents.append(node.sequence.cpu())
                best_log_probs.append(node.log_prob.cpu().item())
                last_scores.append(node.last_score.cpu().item())

        best_sents = torch.stack(pad_tensors(best_sents, pad_token_id), dim=0)
        best_log_probs = torch.tensor(best_log_probs)
        last_scores = torch.tensor(last_scores)
        return GenerateBeamDecoderOnlyOutput(
            sequences=best_sents,
            last_scores=last_scores,
            sequences_scores=best_log_probs,
        )
