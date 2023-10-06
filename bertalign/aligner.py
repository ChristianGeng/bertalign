import numpy as np

from bertalign.corelib import find_first_search_path
from bertalign.corelib import find_second_search_path
from bertalign.corelib import find_top_k_sents
from bertalign.corelib import first_back_track
from bertalign.corelib import first_pass_align
from bertalign.corelib import get_alignment_types
from bertalign.corelib import second_back_track
from bertalign.corelib import second_pass_align
from bertalign.encoder import Encoder
from bertalign.utils import LANG
from bertalign.utils import clean_text
from bertalign.utils import detect_lang
from bertalign.utils import split_sents


class Bertalign:
    r"""Main Aligner Class."""
    def __init__(
        self,
        src_raw,
        tgt_raw,
        max_align=5,
        top_k=3,
        win=5,
        skip=-0.1,
        margin=True,
        len_penalty=True,
        input_type="raw",
        src_lang=None,
        tgt_lang=None,
        model_name="LaBSE",
    ):
        self.max_align = max_align
        self.top_k = top_k
        self.win = win
        self.skip = skip
        self.margin = margin
        self.len_penalty = len_penalty

        input_types = ["raw", "lines", "tokenized"]
        if input_type not in input_types:
            raise ValueError(
                "Invalid input type '%s'. Expected one of: %s"
                % (input_type, input_types)
            )

        if input_type == "lines":
            # need to split
            src = clean_text(src_raw)
            tgt = clean_text(tgt_raw)
            src_sents = src.splitlines()
            tgt_sents = tgt.splitlines()

            if not src_lang:
                src_lang = detect_lang(src)
            if not tgt_lang:
                tgt_lang = detect_lang(tgt)

        elif input_type == "raw":
            src = clean_text(src_raw)
            tgt = clean_text(tgt_raw)

            if not src_lang:
                src_lang = detect_lang(src)
            if not tgt_lang:
                tgt_lang = detect_lang(tgt)

            src_sents = split_sents(src, src_lang)
            tgt_sents = split_sents(tgt, tgt_lang)

        elif input_type == "tokenized":
            if not src_lang:
                src_lang = detect_lang(src)
            if not tgt_lang:
                tgt_lang = detect_lang(tgt)

            src_sents = src_raw
            tgt_sents = tgt_raw

            if not src_lang:
                src_lang = detect_lang(" ".join(src_sents))
            if not tgt_lang:
                tgt_lang = detect_lang(" ".join(tgt_sents))

        src_num = len(src_sents)
        tgt_num = len(tgt_sents)

        src_lang = LANG.ISO[src_lang]
        tgt_lang = LANG.ISO[tgt_lang]

        print("Source language: {}, Number of sentences: {}".format(src_lang, src_num))
        print("Target language: {}, Number of sentences: {}".format(tgt_lang, tgt_num))


        # transformation takes place in the constructor
        # the model comes from the global scope

        # See other cross-lingual embedding models at
        # https://www.sbert.net/docs/pretrained_models.html
        model = Encoder(model_name)
        print("Embedding source text using {} ...".format(model.model_name))
        src_vecs, src_lens = model.transform(src_sents, max_align - 1)
        print("Embedding target text using {} ...".format(model.model_name))
        tgt_vecs, tgt_lens = model.transform(tgt_sents, max_align - 1)

        char_ratio = np.sum(src_lens[0,]) / np.sum(tgt_lens[0,])

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_sents = src_sents
        self.tgt_sents = tgt_sents
        self.src_num = src_num
        self.tgt_num = tgt_num
        self.src_lens = src_lens
        self.tgt_lens = tgt_lens
        self.char_ratio = char_ratio
        self.src_vecs = src_vecs
        self.tgt_vecs = tgt_vecs

    def align_sents(self):
        r"""Aligner Invokation."""
        print("Performing first-step alignment ...")
        D, I = find_top_k_sents(self.src_vecs[0, :], self.tgt_vecs[0, :], k=self.top_k)
        first_alignment_types = get_alignment_types(2)  # 0-1, 1-0, 1-1
        first_w, first_path = find_first_search_path(self.src_num, self.tgt_num)
        first_pointers = first_pass_align(
            self.src_num, self.tgt_num, first_w, first_path, first_alignment_types, D, I
        )
        first_alignment = first_back_track(
            self.src_num,
            self.tgt_num,
            first_pointers,
            first_path,
            first_alignment_types,
        )

        print("Performing second-step alignment ...")
        second_alignment_types = get_alignment_types(self.max_align)
        second_w, second_path = find_second_search_path(
            first_alignment, self.win, self.src_num, self.tgt_num
        )
        second_pointers = second_pass_align(
            self.src_vecs,
            self.tgt_vecs,
            self.src_lens,
            self.tgt_lens,
            second_w,
            second_path,
            second_alignment_types,
            self.char_ratio,
            self.skip,
            margin=self.margin,
            len_penalty=self.len_penalty,
        )
        second_alignment = second_back_track(
            self.src_num,
            self.tgt_num,
            second_pointers,
            second_path,
            second_alignment_types,
        )

        print(
            "Successfully aligned {} {} sentences to {} {} sentences\n".format(
                self.src_num, self.src_lang, self.tgt_num, self.tgt_lang))
        self.result = second_alignment

    def print_sents(self):
        """Print aligned sentence pairs."""
        for bead in self.result:
            src_line = self._get_line(bead[0], self.src_sents)
            tgt_line = self._get_line(bead[1], self.tgt_sents)
            print(src_line + "\n" + tgt_line + "\n")

    @staticmethod
    def _get_line(bead, lines):
        line = ""
        if len(bead) > 0:
            line = " ".join(lines[bead[0] : bead[-1] + 1])
        return line
