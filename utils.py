import os
from statistics import median
from typing import List

import torch
from torch.nn.functional import cosine_similarity
from torchtext.data import Example
from transformers import PreTrainedTokenizer

from diagnnose.activations.selection_funcs import final_sen_token, intersection
from diagnnose.corpus import Corpus
from diagnnose.models import LanguageModel
from diagnnose.probe import DataLoader, DCTrainer
from diagnnose.syntax.tasks.warstadt_preproc import ENVS
from diagnnose.utils.misc import suppress_print

MODEL_NAMES = [
    "full",
    "full2",
    "full3",
    "full4",
    "full5",
    "no_npi",
    "no_npi2",
    "no_npi3",
    "no_npi4",
    "no_npi5",
    "adverbs",
    "conditional",
    "determiner_negation_biclausal",
    "sentential_negation_biclausal",
    "only",
    "quantifier",
    "questions",
    "simplequestions",
    "superlative",
    "adverbs2",
    "conditional2",
    "determiner_negation_biclausal2",
    "sentential_negation_biclausal2",
    "only2",
    "quantifier2",
    "questions2",
    "simplequestions2",
    "superlative2",
    "adverbs3",
    "conditional3",
    "determiner_negation_biclausal3",
    "sentential_negation_biclausal3",
    "only3",
    "quantifier3",
    "questions3",
    "simplequestions3",
    "superlative3",
]


@suppress_print
def monotonicity_probe(
    model: LanguageModel, corpus: Corpus, probe_config, activation_names=None
):
    if activation_names is None:
        activation_names = [(1, "hx")]

    results = {}

    create_new_activations = True  # not os.path.exists(probe_config["save_dir"])

    # uniform split
    data_loader = DataLoader(
        corpus,
        model=model,
        activations_dir=probe_config["save_dir"],
        activation_names=activation_names,
        train_test_ratio=0.9,
        train_selection_func=final_sen_token,
        create_new_activations=create_new_activations,
    )
    dc_trainer = DCTrainer(
        data_loader,
        **probe_config,
    )
    results["uniform"] = dc_trainer.train()

    # held-out environment split
    for env in ENVS:

        def train_selection_func(_w_idx: int, item: Example) -> bool:
            return item.env != env

        def test_selection_func(_w_idx: int, item: Example) -> bool:
            return item.env == env

        data_loader = DataLoader(
            corpus,
            model=model,
            activations_dir=probe_config["save_dir"],
            activation_names=activation_names,
            train_selection_func=intersection((final_sen_token, train_selection_func)),
            test_selection_func=intersection((final_sen_token, test_selection_func)),
        )
        dc_trainer = DCTrainer(
            data_loader,
            **probe_config,
        )
        results[env] = dc_trainer.train()

    return results


@suppress_print
def median_ranks(
    model: LanguageModel,
    corpus: Corpus,
    tokenizer: PreTrainedTokenizer,
    envs: List[str],
    probe_config,
    activation_names=None,
):
    if activation_names is None:
        activation_names = [(1, "hx")]

    results = {}

    for env in ["all"] + envs:

        def train_selection_func(_w_idx: int, item: Example) -> bool:
            return item.env == env or env == "all"

        data_loader = DataLoader(
            corpus,
            model=model,
            activations_dir=probe_config["save_dir"],
            activation_names=activation_names,
            train_test_ratio=0.9,
            train_selection_func=intersection((final_sen_token, train_selection_func)),
            create_new_activations=(len(envs) == 1),
        )
        probe_config["rank"] = 1
        dc_trainer = DCTrainer(
            data_loader,
            **probe_config,
        )
        dc_results = dc_trainer.train()
        median_rank = dc_median_rank(dc_trainer, model.decoder_w, tokenizer)
        results[env] = median_rank, dc_results

    return results


def dc_median_rank(
    dc_trainer: DCTrainer, decoder: torch.Tensor, tokenizer: PreTrainedTokenizer
) -> int:
    if hasattr(dc_trainer.classifier, "module"):
        de_dc = dc_trainer.classifier.module.state_dict()["classifier.0.weight"]

        output_weights = dc_trainer.classifier.module.state_dict()[
            "classifier.1.weight"
        ]
        de_idx = dc_trainer.data_loader.label_vocab["downward"]

        # If the output weight for DE is negative we should invert the DC weights, as we want to
        # find the most similar decoder vectors of the LM (otherwise we'd find the least similar)
        if output_weights[de_idx] < 0:
            de_dc = -de_dc
    else:
        de_dc = torch.tensor(dc_trainer.classifier.coef_)

    de_sim = cosine_similarity(decoder, de_dc)
    de_sim_order = torch.sort(de_sim, descending=True).indices.tolist()

    npis = [
        "dared",
        "any",
        "anybody",
        "anymore",
        "anyone",
        "anything",
        "anywhere",
        "ever",
        "nor",
        "whatsoever",
        "yet",
    ]
    npi_ids = [tokenizer.convert_tokens_to_ids(w) for w in npis]

    median_rank = median([de_sim_order.index(idx) for idx in npi_ids])

    print(median_rank)

    return median_rank
