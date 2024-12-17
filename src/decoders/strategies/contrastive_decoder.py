from typing import Optional, Union, List, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import nn

from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache

from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from .utils import GenerationStrategy, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel, GenerationMixin
    from transformers.generation.streamers import BaseStreamer


class ContrastiveDecoder(GenerationStrategy):
    """
    Implements contrastive decoding method.
    """

    def __init__(self, config: GenerationConfig, check_config: bool = True):
        super().__init__(config, check_config=check_config)

    @classmethod
    def param_check(cls, config: GenerationConfig):
        if config.num_beams != 1:
            raise ValueError(
                f"{cls.__class__.__name__} can only be used with one beam, but num_beams is {config.num_beams}."
            )
        if config.top_k is None or config.top_k <= 1:
            raise ValueError(
                f"{cls.__class__.__name__} requires a top_k parameter greater than one, but top_k is {config.top_k}."
            )
        if config.do_sample:
            raise ValueError(
                f"{cls.__class__.__name__} cannot be used with sampling."
            )
        if config.penalty_alpha is None or config.penalty_alpha <= 0:
            raise ValueError(
                f"{cls.__class__.__name__} requires a penalty_alpha>0, but penalty_alpha is {config.penalty_alpha}."
            )

    def __call__(decoder,
                 self: Union["PreTrainedModel", "GenerationMixin"],
                 input_ids: torch.LongTensor,
                 logits_processor: Optional[LogitsProcessorList] = None,
                 stopping_criteria: Optional[StoppingCriteriaList] = None,
                 synced_gpus: bool = False,
                 streamer: Optional["BaseStreamer"] = None,
                 gen_args: dict = None,
                 **model_kwargs,
                 ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **contrastive search** and can
        be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`]
            or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        if decoder.config.num_return_sequences > 1:
            raise ValueError(
                "num_return_sequences has to be 1 when doing contrastive search, "
                f"but is {decoder.config.num_return_sequences}."
            )
        if not model_kwargs["use_cache"]:
            raise ValueError("Contrastive search requires `use_cache=True`")
        if self._is_stateful:
            # Just like assisted generation, we need to be able to rollback to a previous state (see comment above)
            raise ValueError(
                f"contrastive search is not supported with stateful models, such as {self.__class__.__name__}"
            )
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        # init values
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        top_k = decoder.config.top_k
        penalty_alpha = decoder.config.penalty_alpha
        pad_token_id = decoder.config._pad_token_tensor
        output_attentions = decoder.config.output_attentions
        output_hidden_states = decoder.config.output_hidden_states
        output_scores = decoder.config.output_scores
        output_logits = decoder.config.output_logits
        return_dict_in_generate = decoder.config.return_dict_in_generate
        sequential = decoder.config.low_memory

        # init attention / hidden states / scores tuples
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        # Create cosine_matrix_mask based on the attention_mask
        cosine_matrix_mask = torch.ones_like(input_ids, dtype=torch.long)
        if self.config.is_encoder_decoder:
            if "decoder_attention_mask" in model_kwargs and model_kwargs["decoder_attention_mask"] is not None:
                cosine_matrix_mask = model_kwargs["decoder_attention_mask"]
        else:
            cosine_matrix_mask = model_kwargs["attention_mask"]
        cosine_matrix_mask = cosine_matrix_mask.repeat_interleave(top_k, dim=0)

        this_peer_finished = False

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # if the first step in the loop, encode all the prefix and obtain: (1) past_key_values;
            # (2) last_hidden_states; (3) logit_for_next_step; (4) update model kwargs for the next step
            if model_kwargs.get("past_key_values") is None or (
                isinstance(model_kwargs["past_key_values"], (Cache, EncoderDecoderCache))
                and model_kwargs["past_key_values"].get_seq_length() == 0
            ):
                # prepare inputs
                model_kwargs["use_cache"] = True
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # encode the given prefix and prepare model inputs; encoder-decoder model process the prefix and save
                # the `encoder_outputs`
                outputs = decoder(
                    **model_inputs, return_dict=True, output_hidden_states=True, output_attentions=output_attentions
                )

                # last decoder hidden states will be used to compute the degeneration penalty (cosine similarity with
                # previous tokens)
                if self.config.is_encoder_decoder:
                    last_hidden_states = outputs.decoder_hidden_states[-1]
                else:
                    last_hidden_states = outputs.hidden_states[-1]

                # next logit for contrastive search to select top-k candidate tokens
                # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for this first iteration
                # (the clone itself is always small)
                # .float() is needed to retain precision for later logits manipulations
                logit_for_next_step = outputs.logits[:, -1, :].clone().float()
                logit_for_next_step = logit_for_next_step.to(input_ids.device)

                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                )

                if not sequential:
                    # Expands model inputs top_k times, for batched forward passes (akin to beam search).
                    _, model_kwargs = self._expand_inputs_for_generation(
                        expand_size=top_k, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
                    )

                past_key_values = model_kwargs.get("past_key_values")
                if past_key_values is None:
                    raise ValueError(
                        f"{self.__class__.__name__} does not support caching and therefore **can't** be used "
                        "for contrastive search."
                    )
                elif (
                    not isinstance(past_key_values[0], (tuple, torch.Tensor))
                    or past_key_values[0][0].shape[0] != batch_size
                ):
                    raise ValueError(
                        f"{self.__class__.__name__} does not have a standard cache format and therefore **can't** be "
                        "used for contrastive search without further modifications."
                    )

            # contrastive_search main logic start:
            # contrastive search decoding consists of two steps: (1) candidate tokens recall; (2) candidate re-rank by
            # degeneration penalty
            processed_logit_for_next_step = logits_processor(input_ids, logit_for_next_step)
            next_probs = nn.functional.softmax(processed_logit_for_next_step, dim=-1)

            top_k_probs, top_k_ids = torch.topk(next_probs, dim=-1, k=top_k)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_logits:
                    raw_logits += (logit_for_next_step,)
                if output_scores:
                    scores += (processed_logit_for_next_step,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # This is needed to properly delete outputs.logits which may be very large for this first iteration
            # Otherwise a reference to outputs.logits is kept all along until after the next call to model.forward()
            del outputs

            if not sequential:
                # Replicates the new past_key_values to match the `top_k` candidates
                past = model_kwargs["past_key_values"]
                # If it is a static cache, modify it in-place layer after layer to save memory
                if isinstance(past, DynamicCache) or (
                    isinstance(past, EncoderDecoderCache) and isinstance(past.self_attention_cache, DynamicCache)
                ):
                    past.batch_repeat_interleave(top_k)
                else:
                    new_key_values = []
                    for layer in past:
                        items = []
                        # item is either the key or the value matrix
                        for item in layer:
                            items.append(item.repeat_interleave(top_k, dim=0))
                        new_key_values.append(tuple(items))

                    past = tuple(new_key_values)

                model_kwargs["past_key_values"] = past

            if sequential:
                all_outputs = []
                for i in range(top_k):
                    # compute the candidate tokens by the language model and collect their hidden_states
                    next_model_inputs = self.prepare_inputs_for_generation(top_k_ids[:, i].view(-1, 1), **model_kwargs)

                    outputs = decoder(
                        **next_model_inputs,
                        return_dict=True,
                        output_hidden_states=True,
                        output_attentions=output_attentions,
                    )
                    if isinstance(outputs["past_key_values"], DynamicCache) or (
                        isinstance(outputs["past_key_values"], EncoderDecoderCache)
                        and isinstance(outputs["past_key_values"].self_attention_cache, DynamicCache)
                    ):
                        # Remove past K-V from output since we don't need to stack later
                        outputs["past_key_values"] = None
                        # Remove last token from past K-V since we don't want to append it at this point
                        model_kwargs["past_key_values"].crop(-1)

                    all_outputs.append(outputs)
                outputs = stack_model_outputs(all_outputs, self.config.get_text_config())

            else:
                # compute the candidate tokens by the language model and collect their hidden_states
                # assembles top_k_ids into batch of size k
                next_model_inputs = self.prepare_inputs_for_generation(top_k_ids.view(-1, 1), **model_kwargs)

                outputs = decoder(
                    **next_model_inputs,
                    return_dict=True,
                    output_hidden_states=True,
                    output_attentions=output_attentions,
                )

            # This is essential to avoid having a last reference to the big past K-V and double the necessary memory
            # in the next loop
            del next_model_inputs

            # name is different for encoder-decoder and decoder-only models
            if self.config.is_encoder_decoder:
                next_hidden = outputs.decoder_hidden_states[-1]
                full_hidden_states = outputs.decoder_hidden_states
            else:
                next_hidden = outputs.hidden_states[-1]
                full_hidden_states = outputs.hidden_states

            # .float() is needed to retain precision for later logits manipulations
            logits = outputs.logits[:, -1, :].float()
            context_hidden = last_hidden_states.repeat_interleave(top_k, dim=0)

            # compute the degeneration penalty and re-rank the candidates based on the degeneration penalty and the
            # model confidence. Keeping `selected_idx` on CPU enables multi-device contrastive search and doesn't
            # introduce (noticeable) slowdowns on single-device runs.
            selected_idx = _ranking_fast(
                context_hidden, next_hidden, top_k_probs, cosine_matrix_mask, penalty_alpha, top_k
            )
            cosine_matrix_mask = torch.cat(
                [cosine_matrix_mask, cosine_matrix_mask.new_ones((cosine_matrix_mask.shape[0], 1))], dim=-1
            )
            selected_idx = selected_idx.to("cpu")

            # This will be used instead of the previous inneficient torch.stack(torch.split())
            augmented_idx = torch.tensor([x + i * top_k for i, x in enumerate(selected_idx)])

            # prepare for the next step: (1) next token_id; (2) past_key_values; (3) last_hidden_states for computing
            # the degeneration penalty; (4) logits for selecting next top-k candidates; (5) selected tokens scores
            # (model confidence minus degeneration penalty); (6) decoder hidden_states
            next_tokens = top_k_ids[range(len(top_k_ids)), selected_idx]
            next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), top_k))
            next_hidden = next_hidden[range(batch_size), selected_idx, :]
            last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)

            next_decoder_hidden_states = ()
            for layer in full_hidden_states:
                layer = torch.stack(torch.split(layer, top_k))[range(batch_size), selected_idx, :]
                next_decoder_hidden_states += (layer,)

            # generate past_key_values cache of only the selected token
            if sequential:
                next_model_input = self.prepare_inputs_for_generation(
                    top_k_ids[:, selected_idx].view(-1, 1), **model_kwargs
                )

                selected_outputs = decoder(
                    **next_model_input,
                    return_dict=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )
                next_past_key_values = selected_outputs["past_key_values"]

            else:
                _, next_past_key_values = self._extract_past_from_model_output(outputs)
                # Do it in-place layer per layer to save memory
                if isinstance(next_past_key_values, DynamicCache) or (
                    isinstance(next_past_key_values, EncoderDecoderCache)
                    and isinstance(next_past_key_values.self_attention_cache, DynamicCache)
                ):
                    next_past_key_values.batch_select_indices(augmented_idx)
                else:
                    new_key_values = []
                    for layer in next_past_key_values:
                        items = []
                        # item is either the key or the value matrix
                        for item in layer:
                            items.append(item[augmented_idx, ...])
                        new_key_values.append(tuple(items))

                    next_past_key_values = tuple(new_key_values)

            logit_for_next_step = torch.stack(torch.split(logits, top_k))[range(batch_size), selected_idx, :]
            logit_for_next_step = logit_for_next_step.to(input_ids.device)

            # Rebuilds the relevant parts of the model output for the selected token, for use in the next iteration
            if self.config.is_encoder_decoder:
                next_step_cross_attentions = ()
                next_step_decoder_attentions = ()
                if output_attentions:
                    for layer in outputs.cross_attentions:
                        layer = torch.stack(torch.split(layer, top_k, dim=0))[range(batch_size), selected_idx, ...]
                        next_step_cross_attentions += (layer,)
                    for layer in outputs.decoder_attentions:
                        layer = torch.stack(torch.split(layer, top_k, dim=0))[range(batch_size), selected_idx, ...]
                        next_step_decoder_attentions += (layer,)
                outputs = Seq2SeqLMOutput(
                    past_key_values=next_past_key_values,
                    decoder_hidden_states=next_decoder_hidden_states,
                    decoder_attentions=next_step_decoder_attentions or None,
                    cross_attentions=next_step_cross_attentions or None,
                )
            else:
                next_step_attentions = ()
                if output_attentions:
                    for layer in outputs.attentions:
                        layer = torch.stack(torch.split(layer, top_k, dim=0))[range(batch_size), selected_idx, ...]
                        next_step_attentions += (layer,)
                outputs = CausalLMOutputWithPast(
                    past_key_values=next_past_key_values,
                    hidden_states=next_decoder_hidden_states,
                    attentions=next_step_attentions or None,
                )
            # contrastive_search main logic end

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # stop when each sentence is finished
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            # Contrastive search works by forward looking at the next token, so we need to exclude it from
            # `past_key_values` to be consistent with the other decoding methods
            if model_kwargs.get("past_key_values") is not None:
                if isinstance(model_kwargs["past_key_values"], DynamicCache) or (
                    isinstance(model_kwargs["past_key_values"], EncoderDecoderCache)
                    and isinstance(model_kwargs["past_key_values"].self_attention_cache, DynamicCache)
                ):
                    model_kwargs["past_key_values"].crop(-1)
                else:
                    past_key_values = []
                    for layer in model_kwargs["past_key_values"]:
                        layer_past_key_values = []
                        for item in layer:
                            layer_past_key_values.append(item[..., :-1, :])
                        past_key_values.append(tuple(layer_past_key_values))
                    model_kwargs["past_key_values"] = tuple(past_key_values)

            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids


def _ranking_fast(
    context_hidden: torch.FloatTensor,
    next_hidden: torch.FloatTensor,
    next_top_k_probs: torch.FloatTensor,
    cosine_matrix_mask: torch.LongTensor,
    alpha: float,
    beam_width: int,
) -> torch.FloatTensor:
    """
    Reranks the top_k candidates based on a degeneration penalty (cosine similarity with previous tokens), as described
    in the paper "A Contrastive Framework for Neural Text Generation". Returns the index of the best candidate for each
    row in the batch.
    """
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1, 2)).squeeze(-1)  # [B*K, S]

    # Penalize cosine_matrix based on the cosine_matrix_mask (ignore padding positions)
    # Using a large negative value for masked positions
    cosine_matrix_mask = cosine_matrix_mask.to(dtype=cosine_matrix.dtype)
    cosine_matrix_mask = (1 - cosine_matrix_mask) * torch.finfo(cosine_matrix.dtype).min
    cosine_matrix = cosine_matrix + cosine_matrix_mask

    degeneration_penalty, _ = torch.max(cosine_matrix, dim=-1)  # [B*K]
    next_top_k_probs = next_top_k_probs.view(-1)  # [B*K]
    contrastive_score = (1.0 - alpha) * next_top_k_probs - alpha * degeneration_penalty
    contrastive_score = torch.stack(torch.split(contrastive_score, beam_width))  # [B, K]
    _, selected_idx = contrastive_score.max(dim=-1)  # [B]
    return selected_idx


def stack_model_outputs(model_outputs: List["ModelOutput"], config: "PretrainedConfig") -> "ModelOutput":
    """
    Stack a list of ModelOutput objects (or its subclasses) along the batch_size dimension. The function infers the
    specific ModelOutput subclass from the list provided.
    """
    if not model_outputs:
        raise ValueError("Input list is empty.")

    # Infer the class from the first object in the list
    model_output_cls = type(model_outputs[0])
    num_hidden_layers = config.get_text_config().num_hidden_layers

    # Ensure all objects are of the same type
    if not all(isinstance(obj, model_output_cls) for obj in model_outputs):
        raise ValueError("All elements in the list should be of the same type.")

    # Helper function to concat tensors or tuples of tensors
    def _concat(data):
        """
        Reverse of `_split` function above.
        """
        if any(data is None for data in data):
            return None
        if isinstance(data[0], torch.Tensor):
            return torch.cat(data, dim=0)
        # New cache format
        elif isinstance(data[0], DynamicCache):
            return DynamicCache.from_batch_splits(data, num_hidden_layers=num_hidden_layers)
        elif isinstance(data[0], EncoderDecoderCache):
            return EncoderDecoderCache.from_batch_splits(data, num_hidden_layers=num_hidden_layers)
        elif isinstance(data[0], tuple):
            # If the elements of the tuple are also tuples (e.g., past_key_values in our earlier example)
            if isinstance(data[0][0], tuple):
                return tuple(
                    tuple(torch.cat([attr[i][j] for attr in data], dim=0) for j in range(len(data[0][0])))
                    for i in range(len(data[0]))
                )
            else:
                return tuple(torch.cat([attr[i] for attr in data], dim=0) for i in range(len(data[0])))
        elif isinstance(data[0], (int, float)):
            # If the elements are integers or floats, return a tensor
            return torch.tensor(data)
        else:
            raise TypeError(f"Unexpected attribute type: {type(data[0])}")

    # Use a dictionary comprehension to gather attributes from all objects and concatenate them
    concatenated_data = {
        k: _concat([getattr(model_output, k) for model_output in model_outputs])
        for k in model_output_cls.__dataclass_fields__.keys()
    }

    # Return a new object of the inferred class with the concatenated attributes
    return model_output_cls(**concatenated_data)