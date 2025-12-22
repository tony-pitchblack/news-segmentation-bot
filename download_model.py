import argparse
import os
import shutil

import wandb


def download_ckpt(*, run_id: str, entity: str, project: str, version: str, artifact: str | None, out: str | None) -> str:
    api = wandb.Api()
    name = artifact or f"{entity}/{project}/model-{run_id}:{version}"
    art = api.artifact(name, type="model")
    art_dir = art.download()
    ckpt_path = os.path.join(art_dir, "model.ckpt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"model.ckpt not found in downloaded artifact: {ckpt_path}")
    if out:
        out = os.path.abspath(out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copy2(ckpt_path, out)
        return out
    return ckpt_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="k4j7vuo7")
    p.add_argument("--entity", default="overfit1010")
    p.add_argument("--project", default="lenta_BiLSTM_F1")
    p.add_argument("--version", default="v0")
    p.add_argument("--artifact", default=None, help="Full artifact name like entity/project/artifact:version")
    p.add_argument("--out", default=None, help="Optional output path to copy model.ckpt to")
    args = p.parse_args()
    download_ckpt(
        run_id=args.run_id,
        entity=args.entity,
        project=args.project,
        version=args.version,
        artifact=args.artifact,
        out=args.out,
    )


if __name__ == "__main__":
    main()

