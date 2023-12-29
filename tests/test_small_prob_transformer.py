import torch
from transformers import GenerationConfig

from decoders.simple.ancestral import SamplingDecoder
from decoders.strategies.sbs_helpers.logits_process import LogitsProcessorList, TemperatureLogitsWarper
from decoders.strategies.stochastic_beam_search import SBSLogitProcessor
from decoders import inject_supervitamined_decoders, StochasticBeamSearchDecoder, toolbox, SmallProbTransformer, \
    SmallProbTransformerConfig

type_t = torch.float64
torch.set_default_dtype(type_t)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = SmallProbTransformerConfig()
model = SmallProbTransformer(config).to(device)
inject_supervitamined_decoders(model)


def test_ancestral_sample():
    torch.manual_seed(43)
    input_ids = torch.tensor([[-1]]).to(device)
    output = model.generate(input_ids,
                            generation_strategy=SamplingDecoder(),
                            generation_config=GenerationConfig(max_new_tokens=200, top_k=0, do_sample=True),
                            num_return_sequences=20,)
    routes = output.sequences[:, 1]
    bins = routes.bincount(minlength=config.real_vocab_size)
    print(f"generated tokens: {output.sequences}")
    print(f"routes: {routes}")
    print(f"bins: {bins}")
    print(f"generated probs: {output.sequences_scores}")



def _test_dev():
    torch.manual_seed(43)
    input_ids = torch.tensor([[-2]]).to(device)
    output = model.generate(input_ids,
                            generation_strategy=StochasticBeamSearchDecoder(), num_beams=2, num_return_sequences=1,
                            length_penalty=0.0,
                            early_stopping=True,
                            # logits_processor=LogitsProcessorList([
                            #     SBSLogitProcessor(num_beams=100, batch_size=1)]),
                            # logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature=0.0001)]),
                            # num_beams=1,
                            do_sample=False, return_dict_in_generate=True,
                            beam_scores_type=type_t,
                            output_scores=True)
    # print(f"generated tokens: {output.sequences}")
    # print(f"routes: {output.sequences[:, 1]}")
    routes_freq = output.sequences[:, 1].bincount(minlength=config.real_vocab_size)
    print(f"routes freq: {routes_freq}, total: {routes_freq.sum()}")
    r_lens = route2length(output.sequences[:, 1])

    for i in range(0, 10):
        print(f"Route {i}, len {route2length(i)}, freq {routes_freq[i]}")

        route_i_seqs = output.sequences[:, 1] == i
        # if i==0:
        #     print(output.sequences[route_i_seqs, :][0:10])
        computed_probs = model.compute_transition_scores(sequences=output.sequences, scores=output.scores,
                                                         normalize_logits=True, beam_indices=output.beam_indices)
        gen_probs = output.sequences_scores
        print(f"probs (generate): {gen_probs[route_i_seqs].mean()}")
        print(f"probs (computed): {computed_probs[route_i_seqs, :].sum(dim=1).mean()}")
        own_probs = toolbox.compute_true_logprobs(model, output.sequences)
        print(f"probs (own): {own_probs[route_i_seqs].mean()}")

    # print(f"probs at step 1: {torch.exp(output.scores[0][0, :])}")
    # print(f"probs at step 2: {torch.exp(output.scores[1][0, :])}")
    # print(f"probs at step 3: {torch.exp(output.scores[2][0, :])}")

    # lets estimate f with monte carlo
    true_f = f_experiment(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).double()).sum() / 10
    print(f"true f: {true_f}")
    f_mc = f_experiment(output.sequences[:, 1].double()).mean()
    print(f"estimated f: {f_mc}")
    error = (f_mc - true_f).abs()
    print(f"error: {error}, relative error: {error / true_f}")

if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())