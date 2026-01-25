import os
import pathlib
from typing import Dict, List, Optional

from PIL import Image
from pdf2image import convert_from_path
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    LayoutLMv3ForSequenceClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

# --------------------
# Config
# --------------------
current_directory = pathlib.Path.cwd()
DATA_ROOT = current_directory / "backend/services/data/doc_classification"
MODEL_NAME = "microsoft/layoutlmv3-base"

labels = ["mrz_td1", "mrz_td3", "certificate"]
label2id: Dict[str, int] = {lbl: i for i, lbl in enumerate(labels)}
id2label: Dict[int, str] = {i: lbl for lbl, i in label2id.items()}

IMG_SIZE = (2480, 3508)  # Width, Height A4 at 300 DPI


# --------------------
# Dataset
# --------------------
class MRZImageDataset(Dataset):
    def __init__(
        self,
        root_dir: pathlib.Path,
        split: str,
        processor,
        allow_empty: bool = False,
    ):
        """
        root_dir: data/mrz
        split: "train" or "val"
        Structure:
          data/mrz/train/mrz_td1/*.pdf
          data/mrz/train/mrz_td3/*.pdf
          data/mrz/train/certificate/*.pdf
          data/mrz/val/mrz_td1/*.pdf
          data/mrz/val/mrz_td3/*.pdf
          data/mrz/val/certificate/*.pdf
        """
        self.root_dir = root_dir
        self.split = split
        self.processor = processor
        self.samples: List[Dict[str, object]] = []

        split_dir = root_dir / split
        for lbl in labels:
            class_dir = split_dir / lbl
            if not class_dir.exists():
                continue

            for pdf_path in class_dir.glob("*.pdf"):
                self.samples.append(
                    {
                        "path": pdf_path,
                        "label": label2id[lbl],
                    }
                )

        if not self.samples and not allow_empty:
            raise RuntimeError(f"No PDFs found under {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_from_pdf(self, path: pathlib.Path) -> Image.Image:
        pages = convert_from_path(str(path), dpi=300)
        if not pages:
            raise RuntimeError(f"Empty or unreadable PDF: {path}")
        img = pages[0].convert("RGB")
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        return img

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        img = self._load_image_from_pdf(sample["path"])

        enc = self.processor(
            images=img,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = sample["label"]
        return item


# --------------------
# Main
# --------------------

def main() -> None:
    print(f"Using DATA_ROOT = {DATA_ROOT}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = MRZImageDataset(DATA_ROOT, "train", processor, allow_empty=False)
    val_ds: Optional[MRZImageDataset]
    try:
        val_ds = MRZImageDataset(DATA_ROOT, "val", processor, allow_empty=True)
        if len(val_ds) == 0:
            val_ds = None
            print("No validation samples found – proceeding without eval dataset.")
    except RuntimeError:
        val_ds = None
        print("No validation split at all – proceeding without eval dataset.")

    training_args = TrainingArguments(
        output_dir="backend/services/outputs/layoutlm_mrz",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=10,
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score

        logits, labels_np = eval_pred
        preds = logits.argmax(-1)
        acc = accuracy_score(labels_np, preds)
        f1 = f1_score(labels_np, preds, average="macro")
        return {"accuracy": acc, "f1_macro": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,
        compute_metrics=compute_metrics if val_ds is not None else None,
        data_collator=default_data_collator,
    )

    trainer.train()

    save_dir = "backend/services/layoutlm_models/layoutlmv3-doc_classification"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"Saved fine-tuned model to {save_dir}")


if __name__ == "__main__":
    main()
