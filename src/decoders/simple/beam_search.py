from typing import Optional, Union, TYPE_CHECKING

import torch

from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList

from .simple_beam_utils import SimpleBeamSearch, BeamSearchNode, pad_tensors, separate_encoder_states
from ..strategies.utils import GenerationStrategy, SampleEncoderDecoderOutput

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel, GenerationMixin


def collate_encoder_states(encoder_states):
    new_dict = encoder_states[0].copy()
    for key in encoder_states[0]:
        if encoder_states[0][key] is not None and isinstance(encoder_states[0][key], torch.Tensor):
            new_dict[key] = torch.stack([encoder_states[i][key].squeeze() for i in range(len(encoder_states))], dim=0)
    if encoder_states[0].get("encoder_outputs") is not None:
        new_dict["encoder_outputs"] = collate_encoder_states(
            [encoder_states[i]["encoder_outputs"] for i in range(len(encoder_states))])
    return new_dict


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
                 **model_kwargs,
                 ):
        r"""
        Simple implementation of Beam Search. May use logit sequences of token ids for models with a language modeling head using **multinomial sampling** and
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
            num_beams (`int`, *optional*, defaults to 5):
                Number of beams to use for generation.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.SampleDecoderOnlyOutput`] or [`~generation.SampleEncoderDecoderOutput`].
        """

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        batch_size, _ = input_ids.shape
        num_beams = self.config.num_beams

        print(f"Running simple beam search. batch: {input_ids.shape[0]}, num_beams: {num_beams}, "
              f"logit_processor: {logits_processor}, stopping_criteria: {stopping_criteria}, ")

        searches = [SimpleBeamSearch(num_beams, model.config.eos_token_id) for _ in range(batch_size)]
        encoder_states = separate_encoder_states(batch_size, model.config.is_encoder_decoder, **model_kwargs)

        # 1. Generate initial hypotheses
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = model(**model_inputs, return_dict=True)
        log_probs, next_candidates = torch.topk(torch.log_softmax(outputs.logits[:, -1, :], dim=-1), 2 * num_beams)
        for b_idx in range(batch_size):
            for j in range(num_beams):
                next_token = next_candidates[b_idx, j]
                new_seq = torch.cat((input_ids[b_idx], next_token.unsqueeze(0)))
                node = BeamSearchNode(searches[b_idx], new_seq, log_probs[b_idx, j], encoder_states[b_idx])
                searches[b_idx].add(node, finished=(next_token == model.config.eos_token_id))

        # 2. Beam search generation loop
        while True:
            # Get the current nodes to expand
            nodes = [n for s in searches for n in s.get_current_beams()]
            if len(nodes) == 0:
                break  # All beams ended in EOS
            input_ids = pad_tensors([node.sequence for node in nodes], model.config.pad_token_id)
            input_ids = torch.stack(input_ids, dim=0)
            new_model_kwargs = collate_encoder_states([node.encoder_state for node in nodes])
            model_inputs = model.prepare_inputs_for_generation(input_ids, **new_model_kwargs)
            outputs = model(**model_inputs, return_dict=True)

            logits = outputs.logits[:, -1, :]
            log_probs, next_candidates = torch.topk(torch.log_softmax(logits, dim=-1),
                                                    2 * num_beams)  # (batch_size, 2 * num_beams)

            for b_idx in range(log_probs.shape[0]):
                for j in range(num_beams):
                    next_token = next_candidates[b_idx, j]
                    beam_log_p = nodes[b_idx].log_p + log_probs[b_idx, j]
                    new_seq = torch.cat((input_ids[b_idx], next_token.unsqueeze(0)))
                    node = BeamSearchNode(nodes[b_idx].search, new_seq, beam_log_p, nodes[b_idx].encoder_state)
                    node.search.add(node, finished=(next_token == model.config.eos_token_id))

            for search in searches:
                search.prune()

            if stopping_criteria(input_ids, None):
                break

        # 3. Return best hypotheses
        best_sents = []
        best_scores = []
        for search in searches:
            for node in search.get_best(self.config.num_return_sequences):
                best_sents.append(node.sequence.cpu())
                best_scores.append(node.log_p.cpu().item())

        best_sents = torch.stack(pad_tensors(best_sents, model.config.pad_token_id), dim=0)
        best_scores = torch.tensor(best_scores)
        return SampleEncoderDecoderOutput(
            sequences=best_sents,
            # scores=best_scores,
            sequences_scores=best_scores,
        )


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from ..__init__ import inject_supervitamined_decoders

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    model.eval()
    inject_supervitamined_decoders(model)
    inputs = tokenizer([  # "translate English to German: What is your name, my dear Friend? I missed you so much",
                          # "translate English to German: How old are you?",
                        # "a b c 1 2 ",
                        "summarize: Lorem ipsum dolor "
                    ],
                        return_tensors="pt", padding=True, truncation=True
                    )
    outputs = model.generate(**inputs,
                             generation_strategy=BeamSearchDecoder(),
                             generation_config=GenerationConfig(max_new_tokens=100, num_beams=5,
                                                                num_return_sequences=5))

    standard_seqs = model.generate(**inputs,
                                   # decoder_input_ids = torch.tensor([[    0,  8410,    15,    51,     3, 15432,   440,   103,   322,    19, 3,     9,  1144,    13]]),
                                   num_beams=5, do_sample=0, max_new_tokens=100, length_penalty=0.0,
                                   early_stopping=True, return_dict_in_generate=True, output_scores=True,
                                   num_return_sequences=5)

    # assert torch.all(outputs.sequences == standard_seqs), f"output: {outputs.sequences}, gold: {standard_seqs}"
    print(f"standard:\n{standard_seqs.sequences}")
    print(f"output:\n{outputs.sequences}")
    for i in standard_seqs.sequences:
        print(list(tokenizer.convert_ids_to_tokens(i)))
    print("outputs:")
    for i in outputs.sequences:
        print(list(tokenizer.convert_ids_to_tokens(i)))
    # print(f"generated text: {tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)}")
    # print(f"generated tokens: {outputs.sequences}")
    print(f"standard logp: {standard_seqs.sequences_scores}")
    print(f"generated logp: {outputs.sequences_scores}")
    # _, recompute_logp = compute_true_logprobs(model, outputs.sequences, encoder_input=inputs)
    # print(f"true logp: {recompute_logp.sum(dim=1)}")
