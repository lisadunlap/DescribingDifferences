import logging
from typing import Dict, List, Tuple

import click
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from serve.utils_llm import get_llm_output
import re
import wandb
from components.evaluator import GPTEvaluator, NullEvaluator
from components.proposer import (
    LLMProposer,
    LLMProposerDiffusion,
    VLMFeatureProposer,
    VLMProposer,
    LLMPairwiseProposerWithQuestion,
    DualSidedLLMProposer,
    LLMOnlyProposer
)
from components.ranker import CLIPRanker, LLMRanker, NullRanker, VLMRanker, LLMOnlyRanker, ClusterRanker


def load_config(config: str) -> Dict:
    base_cfg = OmegaConf.load("configs/base.yaml")
    cfg = OmegaConf.load(config)
    final_cfg = OmegaConf.merge(base_cfg, cfg)
    args = OmegaConf.to_container(final_cfg)
    args["config"] = config
    if args["wandb"]:
        wandb.init(
            project=args["project"],
            entity="clipinvariance",
            name=args["data"]["name"],
            group=f'{args["data"]["group1"]} - {args["data"]["group2"]} ({args["data"]["purity"]})',
            config=args,
        )
    return args


def load_data(args: Dict) -> Tuple[List[Dict], List[Dict], List[str]]:
    data_args = args["data"]

    df = pd.read_csv(f"{data_args['root']}/{data_args['name']}.csv")

    if data_args["subset"]:
        old_len = len(df)
        df = df[df["subset"] == data_args["subset"]]
        print(
            f"Taking {data_args['subset']} subset (dataset size reduced from {old_len} to {len(df)})"
        )

    dataset1 = df[df["group_name"] == data_args["group1"]].to_dict("records")
    dataset2 = df[df["group_name"] == data_args["group2"]].to_dict("records")
    group_names = [data_args["group1"], data_args["group2"]]

    if data_args["purity"] < 1:
        logging.warning(f"Purity is set to {data_args['purity']}. Swapping groups.")
        assert len(dataset1) == len(dataset2), "Groups must be of equal size"
        n_swap = int((1 - data_args["purity"]) * len(dataset1))
        dataset1 = dataset1[n_swap:] + dataset2[:n_swap]
        dataset2 = dataset2[n_swap:] + dataset1[:n_swap]
    return dataset1, dataset2, group_names


reduction_prompt_template = """Below is a list of properties, several of which are similar. Please reduce this to a list of the 25 most common distinct properties, ordered by frequency of occurrence.

{properties}

Only respond with a numbered list of properties, no other text.
"""

def reduce_hypotheses(hypotheses: List[str]) -> List[str]:
    reduction_prompt = reduction_prompt_template.format(properties="\n".join(hypotheses))
    reduced_hypotheses = get_llm_output(reduction_prompt, model="gpt-4o")
    reduced_hypotheses = [re.sub(r'^\d+\.\s*|\*\s*', '', h.strip()) for h in reduced_hypotheses.split("\n") if h.strip()]
    return reduced_hypotheses

def propose(args: Dict, dataset1: List[Dict], dataset2: List[Dict]) -> List[str]:
    proposer_args = args["proposer"]
    proposer_args["seed"] = args["seed"]
    proposer_args["captioner"] = args["captioner"]

    proposer = eval(proposer_args["method"])(proposer_args)
    hypotheses, logs, images = proposer.propose(dataset1, dataset2)
    if args["wandb"]:
        wandb.log({"logs": wandb.Table(dataframe=pd.DataFrame(logs))})
        wandb.log({"llm_outputs": wandb.Table(dataframe=pd.DataFrame(images))})
        if "images_group_1" in images[0]:
            for i in range(len(images)):
                wandb.log(
                    {
                        f"group 1 images ({dataset1[0]['group_name']})": images[i][
                            "images_group_1"
                        ],
                        f"group 2 images ({dataset2[0]['group_name']})": images[i][
                            "images_group_2"
                        ],
                    }
                )
    print(f"Hypotheses length: {len(hypotheses)}")
    hypotheses = reduce_hypotheses(hypotheses)
    print(f"Reduced hypotheses length: {len(hypotheses)}")
    return hypotheses


def rank(
    args: Dict,
    hypotheses: List[str],
    dataset1: List[Dict],
    dataset2: List[Dict],
    group_names: List[str],
) -> List[str]:
    ranker_args = args["ranker"]
    ranker_args["seed"] = args["seed"]

    ranker = eval(ranker_args["method"])(ranker_args)

    scored_hypotheses = ranker.rerank_hypotheses(hypotheses, dataset1, dataset2)
    if args["wandb"]:
        table_hypotheses = wandb.Table(dataframe=pd.DataFrame(scored_hypotheses))
        wandb.log({"scored hypotheses": table_hypotheses})
        for i in range(5):
            wandb.summary[f"top_{i + 1}_difference"] = scored_hypotheses[i][
                "hypothesis"
            ].replace('"', "")
            wandb.summary[f"top_{i + 1}_score"] = scored_hypotheses[i]["auroc"]

    if args["evaluator"]["method"] != "NullEvaluator":
        scored_groundtruth = ranker.rerank_hypotheses(
            group_names,
            dataset1,
            dataset2,
        )
        if args["wandb"]:
            table_groundtruth = wandb.Table(dataframe=pd.DataFrame(scored_groundtruth))
            wandb.log({"scored groundtruth": table_groundtruth})

    return [hypothesis["hypothesis"] for hypothesis in scored_hypotheses]


def evaluate(args: Dict, ranked_hypotheses: List[str], group_names: List[str]) -> Dict:
    evaluator_args = args["evaluator"]

    evaluator = eval(evaluator_args["method"])(evaluator_args)

    metrics, evaluated_hypotheses = evaluator.evaluate(
        ranked_hypotheses,
        group_names[0],
        group_names[1],
    )

    if args["wandb"] and evaluator_args["method"] != "NullEvaluator":
        table_evaluated_hypotheses = wandb.Table(
            dataframe=pd.DataFrame(evaluated_hypotheses)
        )
        wandb.log({"evaluated hypotheses": table_evaluated_hypotheses})
        wandb.log(metrics)
    return metrics


@click.command()
@click.option("--config", help="config file")
def main(config):
    logging.info("Loading config...")
    args = load_config(config)
    # print(args)

    logging.info("Loading data...")
    dataset1, dataset2, group_names = load_data(args)
    # print(dataset1, dataset2, group_names)

    logging.info("Proposing hypotheses...")
    hypotheses = propose(args, dataset1, dataset2)
    # print(hypotheses)
    
    logging.info("Ranking hypotheses...")
    ranked_hypotheses = rank(args, hypotheses, dataset1, dataset2, group_names)
    # print(ranked_hypotheses)

    logging.info("Evaluating hypotheses...")
    metrics = evaluate(args, ranked_hypotheses, group_names)
    # print(metrics)


if __name__ == "__main__":
    main()
