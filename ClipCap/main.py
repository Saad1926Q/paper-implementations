from process_images import process_dataset
from dataset_new import CaptionsDataset
from model import ClipCapModel
from train import train_model
from datasets import load_dataset
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    ds = load_dataset("google-research-datasets/conceptual_captions", "unlabeled")

    clip_embs_train, captions_train = process_dataset(ds["train"], max_samples=300000)
    clip_embs_val, captions_val = process_dataset(ds["validation"], max_samples=5000)

    train_ds = CaptionsDataset(clip_embs_train, captions_train, prefix_length=10)
    val_ds = CaptionsDataset(clip_embs_val, captions_val, prefix_length=10)

    model = ClipCapModel(prefix_length=10, clip_length=10, prefix_size=512)
    train_model(dataset=train_ds, val_dataset=val_ds, model=model, output_dir="./checkpoints", output_prefix="coco_prefix", args=args)
