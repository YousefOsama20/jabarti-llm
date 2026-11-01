import argparse
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

REPO = "bakrianoo/jabarti-llm-dataset"
MAX_CHUNKS = 20

HF_FILE ={
    "phase1_train":"pretrain/phase1_train-00000-of-00001.parquet",
    "phasel_eval": "pretrain/phase1_eval-00000-of-00001.parquet" ,

    "phase2_train":"pretrain/phase2_eval-00000-of-00001.parquet" ,
    "phase2_eval": "pretrain/phase2_eval-00000-of-00001.parquet" ,   

    "ft_train": "finetune/train-00000-of-00001.parquet",
    "ft_eval":  "finetune/eval-00000-of-00001.parquet",
}


TRAIN_SPLITS = {"phase1_train", "phase2_train"}
EVAL_SPLITS = {"phase1_eval", "phase2_eval"}

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def downlao_and_filter():

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name , filename in HF_FILE.items():
        local = hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=filename  
       )

        df = pd.read_parquet(local)
        before_filter =len(df)

        if name in TRAIN_SPLITS:

            df = (
                df.sort_values(["article_id","chunk_index"])
                .groupby("article_id",sort=False)
                .head(MAX_CHUNKS)
                .reset_index(drop=True)
            )
        
        after_filter = len(df)

        print(f"df: {name} => {before_filter} filtered to be {after_filter}")
        
        output_path = OUT_DIR / f"{name}_filtered.parquet"
        df.to_parquet(output_path, index= False)
        
        print (f"Saved :{output_path}")


def main():
    downlao_and_filter()

if __name__ == "__main__":
    main()
    