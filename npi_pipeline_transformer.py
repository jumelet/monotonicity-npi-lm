import pickle
import os
from pprint import pprint

from transformers import PreTrainedTokenizer

from diagnnose.config import create_config_dict
from diagnnose.corpus import Corpus
from diagnnose.models import LanguageModel, import_model
from diagnnose.syntax import SyntacticEvaluator
from diagnnose.syntax.tasks.warstadt_preproc import ENVS
from diagnnose.tokenizer import create_tokenizer

from utils import median_ranks, monotonicity_probe


SUPPRESS_PRINT = False
MODEL_NAMES = ["../experiments/andy/checkpoint_best.pt"]

if __name__ == "__main__":
    config_dict = create_config_dict()

    results_dir = config_dict["probe"]["save_dir"]

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    results = {mn: {} for mn in MODEL_NAMES}

    for mn in MODEL_NAMES:
        config_dict["model"]["state_dict"] = mn
        envs = [env for env in ENVS if env == mn[: len(env)]] or ENVS

        model: LanguageModel = import_model(**config_dict["model"])
        tokenizer: PreTrainedTokenizer = create_tokenizer(**config_dict["tokenizer"])
        corpus: Corpus = Corpus.create(tokenizer=tokenizer, **config_dict["corpus"])

        print(f"Probing {mn} on", envs)
        results[mn]["probe"] = monotonicity_probe(
            model, corpus, config_dict["probe"], suppress_print=False, activation_names=[(-1, "hx")]
        )
        print(results[mn]["probe"])

        results[mn]["median_rank"] = median_ranks(
            model,
            corpus,
            tokenizer,
            envs,
            config_dict["probe"],
            suppress_print=SUPPRESS_PRINT,
            activation_names=[(-1, "hx")]
        )
        print("Median rank", results[mn]["median_rank"])

        config_dict["syntax"]["config"]["warstadt"]["subtasks"] = envs
        suite = SyntacticEvaluator(model, tokenizer, **config_dict["syntax"])
        results[mn]["warstadt"] = suite.run()
        print("Syntactic evaluation", results[mn]["warstadt"], "\n\n")

    pprint(results)

    with open(os.path.join(results_dir, "results.pickle"), "wb") as f:
        pickle.dump(results, f)
