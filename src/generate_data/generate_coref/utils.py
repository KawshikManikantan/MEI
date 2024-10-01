import argparse
import os
import re
import collections
from os import path
from coref_data_proc.constants import MODEL_TO_MAX_LEN, MODEL_TO_MODEL_STR
from transformers import LongformerTokenizerFast, AutoTokenizer

BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")
COREF_RESULTS_REGEX = re.compile(
    r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) "
    r"([0-9.]+)%\tF1: ([0-9.]+)%.*",
    re.DOTALL,
)


class BaseDocumentState:
    def __init__(self, key):
        self.doc_key = key
        self.sentence_end = []
        self.token_end = []
        self.tokens = []
        self.subtokens = []
        self.info = []
        self.segments = []
        self.subtoken_map = []
        self.orig_subtoken_map = []
        self.segment_subtoken_map = []
        self.sentence_map = []
        self.pronouns = []
        self.clusters = []
        self.merged_clusters = []
        self.coref_stacks = collections.defaultdict(list)
        self.segment_info = []
        self.speakers = []
        self.cluster_str = []

    def finalize(self):
        raise NotImplementedError


def get_doc_key(doc_id, part):
    return "{}_{}".format(doc_id, int(part))


def normalize_word(word):
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


def flatten(input_list):
    return [item for sublist in input_list for item in sublist]


def split_into_segments(document_state, max_segment_len, constraints1, constraints2):
    current = 0
    while current < len(document_state.subtokens):
        end = min(current + max_segment_len - 1 - 2, len(document_state.subtokens) - 1)
        while end >= current and not constraints1[end]:
            end -= 1
        if end < current:
            end = min(
                current + max_segment_len - 1 - 2, len(document_state.subtokens) - 1
            )
            while end >= current and not constraints2[end]:
                end -= 1
            if end < current:
                raise Exception("Can't find valid segment")
        document_state.segments.append(document_state.subtokens[current : end + 1])
        subtoken_map = document_state.subtoken_map[current : end + 1]
        document_state.segment_subtoken_map.append(subtoken_map)
        if hasattr(document_state, "info"):
            info = document_state.info[current : end + 1]
            document_state.segment_info.append(info)
        current = end + 1


def get_sentence_map(segments, sentence_end):
    current = 0
    sent_map = []
    sent_end_idx = 0
    assert len(sentence_end) == sum([len(s) for s in segments])
    for segment in segments:
        for i in range(len(segment)):
            sent_map.append(current)
            current += int(sentence_end[sent_end_idx])
            sent_end_idx += 1
    return sent_map


def get_tokenizer(model_str):
    if "longformer" in model_str:
        tokenizer = LongformerTokenizerFast.from_pretrained(
            model_str,
            add_prefix_space=True,
            clean_up_tokenization_spaces=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_str,
            clean_up_tokenization_spaces=True,
        )

    return tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Input directory.")
    parser.add_argument("-output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument(
        "-model",
        default="longformer",
        choices=["longformer", "bert", "roberta", "spanbert", "llama"],
        type=str,
        help="Model type.",
    )
    parser.add_argument("-seg_len", default=4096, type=int, help="Max. segment length")
    parser.add_argument(
        "-add_speaker",
        default=False,
        action="store_true",
        help="Speaker represented in text.",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        base_dir = path.dirname(args.input_dir.rstrip("/"))
        args.output_dir = path.join(
            base_dir, args.model + ("_speaker" if args.add_speaker else "")
        )

    print(args.input_dir)
    assert path.exists(args.input_dir)
    assert MODEL_TO_MAX_LEN[args.model] >= args.seg_len

    print(f"Model: {args.model}, Segment length: {args.seg_len}")
    args.model = MODEL_TO_MODEL_STR[args.model]
    args.tokenizer = get_tokenizer(args.model)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    return args


if __name__ == "__main__":
    parse_args()
