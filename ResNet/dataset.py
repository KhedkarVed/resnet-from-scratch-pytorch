import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os


class PPEClassificationDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        with open(ann_file, "r") as f:
            coco = json.load(f)

        # image_id → filename
        self.images = {img["id"]: img["file_name"] for img in coco["images"]}

        # --------------------------------------------------
        # PRIORITY RULE (higher value = higher priority)
        # --------------------------------------------------
        # 1: No helmet
        # 2: No vest
        # 3: Person
        # 4: Helmet
        # 5: Vest
        #
        # Priority logic:
        # If any "No helmet" exists → image = No helmet
        # Else if "No vest" exists → image = No vest
        # Else helmet / vest / person
        # --------------------------------------------------
        priority = {
            1: 5,  # No helmet
            2: 4,  # No vest
            4: 3,  # helmet
            5: 2,  # vest
            3: 1,  # person
        }

        # image_id → selected category_id
        self.labels = {}

        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            cat_id = ann["category_id"]

            # ignore background / invalid category
            if cat_id == 0:
                continue

            if img_id not in self.labels:
                self.labels[img_id] = cat_id
            else:
                # keep higher-priority label
                if priority[cat_id] > priority[self.labels[img_id]]:
                    self.labels[img_id] = cat_id

        # convert labels from 1–5 → 0–4
        for img_id in self.labels:
            self.labels[img_id] -= 1

        self.image_ids = list(self.labels.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        img_path = os.path.join(self.img_dir, self.images[image_id])
        image = Image.open(img_path).convert("RGB")

        label = self.labels[image_id]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label
