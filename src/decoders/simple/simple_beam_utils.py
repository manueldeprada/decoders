from itertools import count
from queue import Queue
from heapq import heappush, heappop
from typing import Dict, Any, List

import torch
from torch.nn.functional import pad


class BSPriorityQueue(Queue):
    """ This PQ has the following customizations for beam search:
     - If there is a tie, FIFO using a counter
     - minheap -> maxheap (by negating scores)
     - gets score from node.eval()
     - defines merge operation
     - defines peek operation
     - defines get_n operation
     """

    def _init(self, maxsize):
        self.queue = []
        self.count = count()

    def _qsize(self):
        return len(self.queue)

    def _put(self, item):
        score = item.eval()
        heappush(self.queue, (-score, next(self.count), item))

    def _get(self):
        score, _, item = heappop(self.queue)
        return item

    def peek(self):
        return self.queue[0][2]

    def get_n(self, n):
        items = []
        for _ in range(n):
            if not self.empty():
                items.append(self.get())
            else:
                break
        return items

    @classmethod
    def merge(cls, q1, q2):
        new = BSPriorityQueue()
        for _ in range(q1.qsize()):
            node = q1.get()
            new.put(node)
        for _ in range(q2.qsize()):
            node = q2.get()
            new.put(node)
        return new

    def __repr__(self):
        return f"BSPriorityQueue(len={len(self.queue)}, data={str(self.queue)})"


class SimpleBeamSearch(object):
    """ Defines a beam search object for a single input sentence. """

    def __init__(self, beam_size, eos_token, eval_by_score=False):
        self.beam_size = beam_size
        self.eos_token = eos_token
        self.eval_by_score = eval_by_score

        self.alive_nodes = BSPriorityQueue()  # beams to be expanded
        self.final_nodes = BSPriorityQueue()  # beams that ended in EOS

    def add(self, node, finished=False):
        """ Adds a new beam search node to the queue of current nodes """
        if node.log_prob == float("-inf"):
            return  # this is the only hack in the algorithm. Discard nodes with -inf log prob. Necessary for toy models
        if finished:
            self.final_nodes.put(node)
        else:
            self.alive_nodes.put(node)

    def get_current_beams(self):
        """ Returns beam_size current nodes with the lowest negative log probability """
        return self.alive_nodes.get_n(self.beam_size)

    def final_best_n(self, number):
        """ Returns final nodes with the lowest negative log probability """
        # Merge EOS paths and those that were stopped by
        # max sequence length (still in nodes)
        merged = BSPriorityQueue.merge(self.final_nodes, self.alive_nodes)
        best = merged.get_n(number)  # todo: make more efficient by peeking both queues instead of merge
        return best

    def get_best_node(self):
        """ Returns and removes from queue the best node. True if final, False otherwise"""
        best_alive = self.alive_nodes.peek() if not self.alive_nodes.empty() else None
        best_final = self.final_nodes.peek() if not self.final_nodes.empty() else None
        best_alive_score = best_alive.eval() if best_alive is not None else float("-inf")
        best_final_score = best_final.eval() if best_final is not None else float("-inf")
        if best_alive is None and best_final is None:
            return None, None
        if (best_alive is None) or (best_alive_score < best_final_score):
            return self.final_nodes.get(), True
        else:
            return self.alive_nodes.get(), False

    def prune(self, keep_k_always_alive=False):
        """ Removes all nodes but the beam_size best ones (lowest neg log prob) """
        new_nodes_active = BSPriorityQueue()
        new_nodes_final = BSPriorityQueue()
        if keep_k_always_alive:  # this is a HACK to mimic HF's early stopping. Should be False for correctness.
            for _ in range(self.beam_size):
                if not self.alive_nodes.empty():
                    node = self.alive_nodes.get()
                    new_nodes_active.put(node)
                if not self.final_nodes.empty():
                    node = self.final_nodes.get()
                    new_nodes_final.put(node)
        else:
            for _ in range(self.beam_size): #TODO: this can be optimized by discarding worse-than-worst nodes on the go
                node, is_final = self.get_best_node()
                if node is None:
                    break
                if not is_final:
                    new_nodes_active.put(node)
                else:
                    new_nodes_final.put(node)
        self.alive_nodes = new_nodes_active
        self.final_nodes = new_nodes_final

    def is_done(self):
        """ Returns whether beam search is complete or not """
        # best_node_final = len(self.final_nodes.peek().sequence) if not self.final_nodes.empty() else 0
        # print((len(self.final_nodes.queue), len(self.alive_nodes.queue),best_node_final ))
        return len(self.final_nodes.queue) >= self.beam_size

    def __repr__(self):
        return f"BSSearch({len(self.alive_nodes.queue)} alive, {len(self.final_nodes.queue)} final)"


class BeamSearchNode(object):
    """ Defines a search node and stores values important for computation of beam search path"""

    def __init__(self, search, sequence, model_state, log_prob, last_score):
        self.search = search
        self.sequence = sequence
        self.model_state = model_state
        self.log_prob = log_prob
        self.last_score = last_score

    def eval(self):
        if self.search.eval_by_score:
            return self.last_score
        else:
            return self.log_prob

    def __repr__(self):
        return f"BeamSearchNode(logp={self.log_prob.item():.2f}, score={self.last_score.item():.2f}, seq={self.sequence})"


# In the following ugly code rests the decent speed of this beam search implementation.
def collate_model_states(model_states, disable_kv_cache=False):
    new_dict = model_states[0].copy()
    for key in model_states[0]:
        if model_states[0][key] is not None and isinstance(model_states[0][key], torch.Tensor):
            new_dict[key] = torch.stack([model_states[i][key].squeeze() for i in range(len(model_states))], dim=0)
    if model_states[0].get("encoder_outputs") is not None:
        new_dict["encoder_outputs"] = collate_model_states(
            [model_states[i]["encoder_outputs"] for i in range(len(model_states))])
    if model_states[0].get("past_key_values") is not None and not disable_kv_cache:
        tt = []
        for i in range(len(model_states[0]["past_key_values"])):
            list_t = []
            for j in range(len(model_states[0]["past_key_values"][i])):
                t = torch.stack([model_states[k]["past_key_values"][i][j] for k in range(len(model_states))], dim=0)
                list_t.append(t)
            tt.append(tuple(list_t))
        new_dict["past_key_values"] = tuple(tt)
    if disable_kv_cache:
        new_dict.pop("past_key_values", None)
    return new_dict


def update_model_kv_cache(model_state, index, model_kwargs, disable_kv_cache=False):
    new_dict = model_state.copy()
    if not disable_kv_cache:
        new_dict["past_key_values"] = tuple(
            [tuple([a[index] for a in b]) for b in model_kwargs["past_key_values"]])
    else:
        new_dict.pop("past_key_values", None)
    return new_dict


def separate_model_states(
        batch_size: int,
        is_encoder_decoder: bool = False,
        disable_kv_cache: bool = False,
        new_method=False,
        **model_kwargs,
) -> List[Dict[str, Any]]:
    """  Separate the batch of encoder states into a list of individual states. """

    def _separate_dict(dict_to_expand, i):
        new_dict = dict_to_expand.copy()
        for key in dict_to_expand:
            if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                new_dict[key] = dict_to_expand[key][i].unsqueeze(0)
        return new_dict

    model_kwargs_l = [_separate_dict(model_kwargs, i) for i in range(batch_size)]

    if is_encoder_decoder:
        if model_kwargs.get("encoder_outputs") is None:
            raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
        for i in range(batch_size):
            model_kwargs_l[i]["encoder_outputs"] = _separate_dict(model_kwargs["encoder_outputs"], i)
        if model_kwargs.get("past_key_values") is not None and not disable_kv_cache:
            for i in range(batch_size):
                if new_method:
                    pass #recursive_replace_tensors(model_kwargs_l[i]["past_key_values"], i)
                else:
                    model_kwargs_l[i]["past_key_values"] = tuple(
                        [tuple([a[i] for a in b]) for b in model_kwargs["past_key_values"]])
    return model_kwargs_l


def pad_tensors(tensors, pad_token_id):
    if len(tensors[0].shape) > 0:
        max_len = max([t.shape[0] for t in tensors])
        tensors = [pad(t, (0, max_len - t.shape[0]), value=pad_token_id) for t in tensors]
        return tensors
    raise ValueError("Tensors must have at least two dimensions")
