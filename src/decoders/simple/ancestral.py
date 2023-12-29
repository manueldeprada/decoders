from typing import Optional, Union, TYPE_CHECKING

import torch
from torch import nn

from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList

from ..strategies.utils import GenerationStrategy, SampleEncoderDecoderOutput

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel, GenerationMixin


class SamplingDecoder(GenerationStrategy):

    def __init__(self, config: GenerationConfig = None, check_config: bool = True):
        super().__init__(config, check_config=check_config)

    @classmethod
    def param_check(cls, config: GenerationConfig):
        pass

    def __call__(self,
                 model: Union["PreTrainedModel", "GenerationMixin"],
                 input_ids: torch.LongTensor,
                 logits_processor: Optional[LogitsProcessorList] = None,
                 logits_warper: Optional[LogitsProcessorList] = None,
                 stopping_criteria: Optional[StoppingCriteriaList] = None,
                 **model_kwargs,
                 ):
        r"""
        Generates ancestral sequences. May use logit sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.SampleDecoderOnlyOutput`] or [`~generation.SampleEncoderDecoderOutput`].
"""

        # 12. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=self.config.num_return_sequences,
            is_encoder_decoder=model.config.is_encoder_decoder,
            **model_kwargs,
        )
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        logits_warper = logits_warper if logits_warper is not None else model._get_logits_warper(self.config)
        eos_token_id_tensor = torch.tensor([model.config.eos_token_id]).to(input_ids.device)

        # init attention / hidden states / scores tuples
        scores = () if self.config.output_scores else None

        # keep track of which sequences are already finished
        sequences_logscores = torch.zeros(input_ids.shape[0], device=input_ids.device)
        finished = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.bool)

        print(f"Running simple ancestral sampling. Batch Size: {input_ids.shape[0]}, "
              f"logit_processor: {logits_processor}, stopping_criteria: {stopping_criteria}, "
              f"logits_warper: {logits_warper}")
        while True:
            model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = model(
                **model_inputs,
                return_dict=True,
            )
            next_token_logits = outputs.logits[:, -1, :]

            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            if scores is not None:
                scores += (next_token_scores,)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1) #todo: test and replace this with gumbel sampling
            next_token_logprobs = next_token_scores.log_softmax(dim=-1).gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)
            # next_token_logprobs = torch.where(~finished, next_token_logprobs, torch.zeros_like(next_token_logprobs))
            sequences_logscores += next_token_logprobs * (~finished).float()


            # finished sentences should have their next token be a padding token
            if model.config.eos_token_id is not None:
                next_tokens = next_tokens * ~finished + model.config.pad_token_id * finished

            finished = finished | (next_tokens == model.config.eos_token_id)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                # unfinished_sequences = unfinished_sequences.mul(
                #     next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                # )

                # stop when each sentence is finished
                if (~finished).max() == 0:
                    break

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                break

        return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                sequences_scores=sequences_logscores,
            )

