import os
import time
import io
import logging
import requests
from urllib.request import urlopen
from urllib.parse import urlparse
from contextlib import suppress
from typing import Tuple, Optional, List, Callable, Dict, Any, Deque
from io import BytesIO

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import numpy as np 

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): 
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    )

def no_drift_fn(data: Any, **kwargs) -> Any:
    """A pass-through function for when no drift is applied."""
    return data

def image_drift_degradation(
    image: Image.Image,
    jpeg_quality: int = 50, # Lower is more degradation (0-100)
    noise_std: float = 0.05,  # Standard deviation of Gaussian noise (0-1 scale)
    blur_radius: Optional[float] = None, # Optional Gaussian blur radius
    **kwargs
) -> Image.Image:
    """
    Simulates image degradation by applying JPEG compression and/or Gaussian noise/blur.
    """
    drifted_image = image.copy() 

    # 1. Optional Blur
    if blur_radius is not None and blur_radius > 0:
        drifted_image = drifted_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        logger.debug(f"Applied blur with radius {blur_radius}")

    # 2. JPEG Compression
    if 0 < jpeg_quality < 100:
        buffer = io.BytesIO()
        drifted_image.save(buffer, format="JPEG", quality=jpeg_quality)
        buffer.seek(0)
        drifted_image = Image.open(buffer).convert("RGB")
        logger.debug(f"Applied JPEG compression with quality {jpeg_quality}")

    # 3. Gaussian Noise (if std > 0)
    if noise_std > 0:
        try:
            img_array = np.array(drifted_image).astype(np.float32) / 255.0
            noise = np.random.normal(loc=0.0, scale=noise_std, size=img_array.shape)
            noisy_img_array = np.clip(img_array + noise, 0.0, 1.0)
            drifted_image = Image.fromarray((noisy_img_array * 255).astype(np.uint8))
            logger.debug(f"Applied Gaussian noise with std {noise_std}")
        except Exception as e:
            logger.error(f"Error applying noise to image: {e}")

    return drifted_image


def text_drift_semantic_shift(
    text: str,
    class_label: Optional[int] = None, # For class-conditional drift
    target_class_for_drift: Optional[int] = None, # Apply only if class_label matches this
    keyword_map: Optional[Dict[str, str]] = None,
    append_phrase: Optional[str] = None,
    **kwargs
) -> str:
    """
    Simulates semantic shift in text by replacing keywords or appending phrases,
    potentially conditioned on the class label.
    """
    modified_text = text

    if target_class_for_drift is not None and class_label != target_class_for_drift:
        return text # No drift if not the target class

    if keyword_map:
        logger.debug(f"Applying keyword map for semantic shift: {keyword_map}")
        for old, new in keyword_map.items():
            # Using case-insensitive replacement for broader match
            # For precise match, remove re.IGNORECASE or use simple string.replace
            import re
            modified_text = re.sub(r'\b' + re.escape(old) + r'\b', new, modified_text, flags=re.IGNORECASE)


    if append_phrase:
        logger.debug(f"Appending phrase for semantic shift: {append_phrase}")
        modified_text = f"{modified_text} {append_phrase}"
        
    return modified_text


def pil_collate_fn(
    batch: List[Tuple[str, Image.Image, torch.Tensor, Dict[str, Any]]]
) -> Tuple[List[str], List[Image.Image], torch.Tensor, List[Dict[str, Any]]]:
    """
    Collates batch items. Keeps PIL images as a list, text as a list of strings.
    Labels are tensorized. Includes original_data_dicts.
    """
    texts, images, labels, original_data_dicts = zip(*batch)
    labels_t = torch.stack(labels) 
    return list(texts), list(images), labels_t, list(original_data_dicts)


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn=pil_collate_fn,
) -> DataLoader:
    """
    Standard DataLoader builder with DistributedSampler support.
    """
    sampler = None
    use_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    
    dataloader_shuffle = shuffle
    dataloader_drop_last = drop_last # Pass original drop_last to DataLoader if not DDP

    if use_distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        dataloader_shuffle = False # Sampler handles shuffle
        # drop_last is handled by sampler, so DataLoader should not also drop_last to avoid losing extra samples
        dataloader_drop_last = False 
        logger.info(
            f"Using DistributedSampler (rank={torch.distributed.get_rank()}) for dataset of len {len(dataset)}."
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataloader_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=dataloader_drop_last, 
        collate_fn=collate_fn,
    )

def fix_imgur_url(url: str) -> str:
    """
    Convert Imgur page URLs to direct image links on i.imgur.com ending in .jpg
    """
    lower = url.lower()
    if "imgur.com" in lower and not lower.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
        return url.replace("imgur.com", "i.imgur.com") + ".jpg"
    return url

class LabeledDataset(Dataset):
    """
    Dataset class that accepts a pre-loaded pandas DataFrame.
    Handles image loading and on-the-fly drift application based on provided functions.
    """
    def __init__(
        self,
        # General config
        config: Dict[str, Any], 
        initial_df: pd.DataFrame, 
        # Drift simulation parameters
        image_drift_fn: Optional[Callable[[Image.Image, Any], Image.Image]] = no_drift_fn,
        text_drift_fn: Optional[Callable[[str, Any], str]] = no_drift_fn, # Takes text, label, args
        image_drift_args: Optional[Dict[str, Any]] = None,
        text_drift_args: Optional[Dict[str, Any]] = None,
        apply_drift_to_specific_class: Optional[int] = None,
    ):
        super().__init__()
        self.config = config 
        
        if initial_df is None:
            raise ValueError("LabeledDataset must be initialized with a pandas DataFrame via 'initial_df'.")
        
        self.metadata = initial_df.copy().reset_index(drop=True)
        logger.info(f"Initialized LabeledDataset with a DataFrame of {len(self.metadata)} rows.")

        if len(self.metadata) == 0:
            logger.warning("LabeledDataset initialized with empty metadata!")

        # Ensure essential columns exist
        required_cols = {"text", "label"}
        if not required_cols.issubset(self.metadata.columns):
            missing = required_cols - set(self.metadata.columns)
            raise ValueError(f"Provided DataFrame is missing required columns: {missing}")
        if "image_path" not in self.metadata.columns and "image_url" not in self.metadata.columns:
            raise ValueError("Provided DataFrame is missing 'image_path' or 'image_url' column.")

        self.image_drift_fn = image_drift_fn if image_drift_fn else no_drift_fn
        self.text_drift_fn = text_drift_fn if text_drift_fn else no_drift_fn # Expects fn(text, class_label, **args)
        self.image_drift_args = image_drift_args if image_drift_args is not None else {}
        self.text_drift_args = text_drift_args if text_drift_args is not None else {}
        self.apply_drift_to_specific_class = apply_drift_to_specific_class


    def __len__(self) -> int:
        return len(self.metadata)

    def load_image(self, source: str) -> Optional[Image.Image]:
        """
        Fetch PIL image from URL or local path, return None on failure.
        """
        if source.startswith(('http://', 'https://')):
            source = fix_imgur_url(source)
            headers = {"User-Agent": "Mozilla/5.0 (compatible)"}
            for attempt in range(3):
                try:
                    resp = requests.get(source, timeout=10, headers=headers)
                    resp.raise_for_status()
                    content_type = resp.headers.get("Content-Type", "")
                    if not content_type.startswith("image/"):
                        raise ValueError(f"URL did not return an image (Content-Type: {content_type})")
                    return Image.open(BytesIO(resp.content)).convert("RGB")
                except Exception as e:
                    status = getattr(e, 'response', None)
                    logger.error(f"Error loading image [{source}], attempt {attempt}: {e}")
                    if hasattr(e, 'response') and getattr(e.response, 'status_code', None) == 429 and attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    break
            return None
        else:
            try:
                return Image.open(source).convert('RGB')
            except Exception as e:
                logger.error(f"Error loading local image [{source}]: {e}")
                return None

    def _open_image(self, row: pd.Series) -> Image.Image:
        """
        Try loading via self.load_image for both local path and URL,
        falling back to a gray placeholder if both fail.
        """
        import os
        import pandas as pd
        from PIL import Image
        from io import BytesIO

        data_root = self.config.get("data_root", ".")
        folder    = self.config.get("dataset_folder_name", "")

        # 1) Try local path
        img_path = row.get("image_path")
        if pd.notna(img_path) and img_path:
            full_path = str(img_path)
            if not os.path.isabs(full_path):
                full_path = os.path.join(data_root, folder, full_path)
            img = self.load_image(full_path)
            if img is not None:
                return img

        # 2) Try URL
        img_url = row.get("image_url")
        if pd.notna(img_url) and img_url:
            img = self.load_image(str(img_url))
            if img is not None:
                return img

        # 3) Fallback placeholder
        row_id = getattr(row, "name", "N/A")
        logger.warning(
            f"[ _open_image ] Could not load image for row {row_id}. "
            f"Tried path='{img_path}', url='{img_url}'. Returning gray placeholder."
        )
        return Image.new("RGB", (224, 224), color=(127, 127, 127))


    def __getitem__(
        self, idx: int
    ) -> Tuple[str, Image.Image, torch.Tensor, Dict[str, Any]]:
        if not 0 <= idx < len(self.metadata):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self.metadata)}")
        row = self.metadata.iloc[idx]
        
        original_text = str(row["text"])
        original_img = self._open_image(row)
        label_val = int(row["label"]) # Assuming labels are integer class indices
        label = torch.tensor(float(label_val), dtype=torch.float32) # Still float for BCELoss

        text_to_return = original_text
        img_to_return = original_img
        
        applied_drift_info = "no_drift_applied"
        original_data_for_reference = {"applied_drift_info": applied_drift_info} # Default

        # Check if drift should be applied to this item
        # Drift is applied if the instance of LabeledDataset was initialized with active drift functions
        is_drift_candidate = (self.image_drift_fn != no_drift_fn) or \
                             (self.text_drift_fn != no_drift_fn)
        
        apply_drift_this_item = is_drift_candidate
        if is_drift_candidate and self.apply_drift_to_specific_class is not None and \
           label_val != self.apply_drift_to_specific_class:
            apply_drift_this_item = False # Don't drift if not the target class

        if apply_drift_this_item:
            img_to_process = original_img 
            
            # Pass class label to drift functions if they need it for conditional drift
            current_text_drift_args = self.text_drift_args.copy()
            current_text_drift_args['class_label'] = label_val
            current_image_drift_args = self.image_drift_args.copy()
            current_image_drift_args['class_label'] = label_val

            img_to_return = self.image_drift_fn(img_to_process, **current_image_drift_args)
            text_to_return = self.text_drift_fn(original_text, **current_text_drift_args)
            
            drift_types_applied = []
            # Check if functions are not no_drift_fn before claiming drift
            if self.image_drift_fn != no_drift_fn : drift_types_applied.append("image")
            if self.text_drift_fn != no_drift_fn : drift_types_applied.append("text")
            
            if drift_types_applied:
                applied_drift_info = f"{'_and_'.join(drift_types_applied)}_drift_applied_to_class_{label_val}"
                original_data_for_reference.update({
                    "original_text_preview": original_text[:60] + ("..." if len(original_text) > 60 else ""),
                    "original_label": label_val,
                    "applied_drift_info": applied_drift_info
                })
            else: # This case implies drift_fns were no_drift_fn, so ensure info is accurate
                 applied_drift_info = "drift_configured_but_fns_were_no_drift"
                 original_data_for_reference["applied_drift_info"] = applied_drift_info

        return text_to_return, img_to_return, label, original_data_for_reference


class UnlabeledDataset(Dataset):
    def __init__(
        self,
        config: Dict[str, Any],
        initial_df: pd.DataFrame, 
    ):
        super().__init__()
        self.config = config

        if initial_df is None:
            raise ValueError("UnlabeledDataset must be initialized with a pandas DataFrame via 'initial_df'.")

        self.metadata = initial_df.copy().reset_index(drop=True)
        logger.info(f"Initialized UnlabeledDataset with a DataFrame of {len(self.metadata)} rows.")

        if len(self.metadata) == 0:
            logger.warning("UnlabeledDataset initialized with empty metadata!")

        required_cols = {"text"}
        if not required_cols.issubset(self.metadata.columns):
            missing = required_cols - set(self.metadata.columns)
            raise ValueError(f"Unlabeled DataFrame missing required columns: {missing}")
        if "image_path" not in self.metadata.columns and "image_url" not in self.metadata.columns:
            raise ValueError("Unlabeled DataFrame missing 'image_path' or 'image_url' column.")

        # For image loading, reusing LabeledDataset's _open_image via a helper instance
        # This is a pragmatic choice for the deadline.
        self._labeled_dataset_img_loader_ref = LabeledDataset(
            config=config, initial_df=pd.DataFrame() 
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def _open_image(self, row: pd.Series) -> Image.Image:
        return self._labeled_dataset_img_loader_ref._open_image(row)

    def __getitem__(
        self, idx: int
    ) -> Tuple[str, Image.Image, torch.Tensor, Dict[str, Any]]:
        if not 0 <= idx < len(self.metadata):
            raise IndexError(f"Index {idx} out of bounds for UnlabeledDataset of length {len(self.metadata)}")
        row = self.metadata.iloc[idx]
        
        text = str(row["text"])
        img = self._open_image(row)
        
        dummy_label = torch.tensor(-1.0, dtype=torch.float32) 
        meta_info = {"applied_drift_info": "unlabeled_data_no_drift_applied"}
        
        return text, img, dummy_label, meta_info
