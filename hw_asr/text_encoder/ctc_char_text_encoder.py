from typing import List, NamedTuple

import numpy as np
import torch

from .char_text_encoder import CharTextEncoder

from pyctcdecode import build_ctcdecoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        lm_vocab = [''] + list(self.alphabet)
        print(self.alphabet)
        print(len(self.alphabet))
        self.decoder = build_ctcdecoder(
            lm_vocab,
            alpha=0.5,
            beta=1.0,
            unigrams=vocab
        )

    def ctc_decode(self, inds: List[int]) -> str:
        decoded_text = []
        prev_char = 0
        for i in inds:
            if prev_char == i:
                continue
            elif i == 0:
                prev_char = i
                continue
            else:
                decoded_text.append(self.ind2char[i])
                prev_char = i
        return ''.join(decoded_text)

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []

        probs = np.array(probs)

        prefix = {('', self.EMPTY_TOK): 1}

        for i in range(probs_length):

            new_prefix = {}
            for (curr_decode, last_char), curr_prob in prefix.items():
                for ind in self.ind2char.keys():
                    char = self.ind2char[ind]
                    if char == last_char:
                        next_decode = curr_decode
                        next_char = last_char
                    elif char == self.EMPTY_TOK:
                        next_decode = curr_decode
                        next_char = self.EMPTY_TOK
                    else:
                        next_decode = curr_decode + char
                        next_char = char

                    next_prob = curr_prob * probs[i][ind]

                    new_prefix[(next_decode, next_char)] = next_prob

            hypos_list = list(new_prefix.items())

            hypos_list.sort(key=lambda x: x[1], reverse=True)

            prefix = dict(hypos_list[:beam_size])

        for hypo in prefix.items():
            hypos.append(Hypothesis(hypo[0][0], hypo[1]))

        return sorted(hypos, key=lambda x: x.prob, reverse=True)

    def ctc_lm_beam_search(self, probs: torch.tensor, probs_length,
                                  beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search with LM and returns a list of pairs (hypothesis, hypothesis probability).
        """
        logits = probs[:probs_length].numpy()
        text = self.decoder.decode(logits, beam_size)

        return [Hypothesis(text, 1)]
