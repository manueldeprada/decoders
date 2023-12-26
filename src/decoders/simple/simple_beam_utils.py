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

    def __init__(self, beam_size, eos_token):
        self.beam_size = beam_size
        self.eos_token = eos_token

        self.alive_nodes = BSPriorityQueue()  # beams to be expanded
        self.final_nodes = BSPriorityQueue()  # beams that ended in EOS

    def add(self, node, finished=False):
        """ Adds a new beam search node to the queue of current nodes """
        if finished:
            self.final_nodes.put(node)
        else:
            self.alive_nodes.put(node)

    def get_current_beams(self):
        """ Returns beam_size current nodes with the lowest negative log probability """
        return self.alive_nodes.get_n(self.beam_size)

    def get_best(self, number):
        """ Returns final nodes with the lowest negative log probability """
        # Merge EOS paths and those that were stopped by
        # max sequence length (still in nodes)
        merged = BSPriorityQueue.merge(self.final_nodes, self.alive_nodes)
        best = merged.get_n(number)  # todo: make more efficient by peeking both queues instead of merge
        return best

    def prune(self):
        """ Removes all nodes but the beam_size best ones (lowest neg log prob) """
        nodes = BSPriorityQueue()
        # Keep track of how many search paths are already finished (EOS)
        finished = self.final_nodes.qsize()
        for _ in range(self.beam_size):# - finished):
            node = self.alive_nodes.get()
            nodes.put(node)
        self.alive_nodes = nodes

    def __repr__(self):
        return f"BSSearch({len(self.alive_nodes.queue)} alive, {len(self.final_nodes.queue)} final)"


class BeamSearchNode(object):
    """ Defines a search node and stores values important for computation of beam search path"""

    def __init__(self, search, sequence, logProb, encoder_state):
        self.search = search
        self.sequence = sequence
        self.log_p = logProb
        self.encoder_state = encoder_state

    def eval(self):
        return self.log_p

    def __repr__(self):
        return f"BeamSearchNode({self.log_p.item():.2f}, {self.sequence})"

def separate_encoder_states(
        batch_size: int,
        is_encoder_decoder: bool = False,
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

    return model_kwargs_l


def pad_tensors(tensors, pad_token_id):
        if len(tensors[0].shape) > 0:
            max_len = max([t.shape[0] for t in tensors])
            tensors = [pad(t, (0, max_len - t.shape[0]), value=pad_token_id) for t in tensors]
            return tensors
        raise ValueError("Tensors must have at least two dimensions")