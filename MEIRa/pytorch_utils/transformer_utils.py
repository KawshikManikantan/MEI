from transformers import LongformerTokenizerFast, AutoTokenizer, PreTrainedTokenizerFast


def get_tokenizer(model_str: str) -> PreTrainedTokenizerFast:
    if "longformer" in model_str:
        tokenizer = LongformerTokenizerFast.from_pretrained(
            model_str,
            add_prefix_space=True,
            clean_up_tokenization_spaces=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_str, use_fast=True, clean_up_tokenization_spaces=True
        )

    return tokenizer
