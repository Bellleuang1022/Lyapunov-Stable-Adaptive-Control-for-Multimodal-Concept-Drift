import logging
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import os 
from typing import List, Tuple, Dict, Optional, Any 
from PIL import Image

try:
    from transformers import (
        AutoModel,
        AutoTokenizer,
        ViTModel,
        ViTImageProcessor,
        AutoProcessor, 
        CLIPTextModel,
        CLIPImageProcessor,
        CLIPVisionModel
    )
except ImportError:
    logging.error("Transformers library not found. Please install it: pip install transformers")
    raise

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): 
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    )


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, device: torch.device, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.config_args = config if config else {} # Store config if needed for other params

        logger.info(f"Initializing TextEncoder with model: {self.model_name}")
        try:
            # self.backbone = AutoModel.from_pretrained(self.model_name)
            self.backbone = CLIPTextModel.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            logger.error(f"Error loading text model/tokenizer {self.model_name}: {e}")
            raise

        hidden_size = self.backbone.config.hidden_size
        head_hidden_dim = max(1, hidden_size // 2) if self.config_args.get("text_head_complex", False) else hidden_size
        
        if self.config_args.get("text_head_complex", False):
            self.head = nn.Sequential(
                nn.Linear(hidden_size, head_hidden_dim),
                nn.ReLU(),
                nn.Linear(head_hidden_dim, 1),
            )
        else: 
            self.head = nn.Linear(hidden_size, 1)
        
        self.to(device) # Move model to device

    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters: texts (List[str]): Batch of input texts.
        Returns:
            emb (torch.Tensor): Embeddings, shape [B, H_text].
            prob (torch.Tensor): Output probabilities, shape [B, 1].
        """
        if not texts: # Handle empty input case
            logger.warning("TextEncoder received empty list of texts.")
            dummy_emb = torch.empty(0, self.backbone.config.hidden_size, device=self.device)
            dummy_prob = torch.empty(0, 1, device=self.device)
            return dummy_emb, dummy_prob

        try:
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length
            )
        except Exception as e:
            logger.error(f"Error during tokenization: {e}. Texts sample: {texts[:2]}")
            raise
            
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.backbone(**inputs)
        # Use CLS token embedding (standard for sentence-level tasks)
        emb = outputs.last_hidden_state[:, 0] 
        
        logits = self.head(emb)
        prob = torch.sigmoid(logits)
        return emb, prob


class ImageEncoder(nn.Module):
    def __init__(self, model_name: str, device: torch.device, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.config_args = config if config else {}

        logger.info(f"Initializing ImageEncoder with model: {self.model_name}")
        try:
            self.backbone = CLIPVisionModel.from_pretrained(self.model_name)
            # Try ViTImageProcessor first, fallback to AutoProcessor
            try:
                self.processor = CLIPImageProcessor.from_pretrained(self.model_name)
            except OSError:
                logger.warning(f"ViTImageProcessor not found for {self.model_name}, trying AutoProcessor.")
                self.processor = AutoProcessor.from_pretrained(self.model_name)
        except Exception as e:
            logger.error(f"Error loading image model/processor {self.model_name}: {e}")
            raise

        hidden_size = self.backbone.config.hidden_size
        head_hidden_dim = max(1, hidden_size // 2) if self.config_args.get("image_head_complex", False) else hidden_size

        if self.config_args.get("image_head_complex", False):
            self.head = nn.Sequential(
                nn.Linear(hidden_size, head_hidden_dim),
                nn.ReLU(),
                nn.Linear(head_hidden_dim, 1),
            )
        else: 
            self.head = nn.Linear(hidden_size, 1)

        self.to(device) # Move model to device

    def forward(self, images: List[Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters: images (List[PIL.Image.Image]): Batch of input images.
        Returns:
            emb (torch.Tensor): Embeddings, shape [B, H_image].
            prob (torch.Tensor): Output probabilities, shape [B, 1].
        """
        if not images: # Handle empty input case
            logger.warning("ImageEncoder received empty list of images.")
            dummy_emb = torch.empty(0, self.backbone.config.hidden_size, device=self.device)
            dummy_prob = torch.empty(0, 1, device=self.device)
            return dummy_emb, dummy_prob
        
        try:
            inputs = self.processor(images=images, return_tensors="pt")
        except Exception as e:
            logger.error(f"Error during image processing: {e}. Number of images: {len(images)}")
            raise

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.backbone(**inputs)
        emb = outputs.last_hidden_state[:, 0] # CLS token for ViT
        
        logits = self.head(emb)
        prob = torch.sigmoid(logits)
        return emb, prob


class WeightedAverage(nn.Module):
    """
    Learnable scalar alpha ∈ [0,1]  →  p = (1-alpha)·p_text + alpha·p_image
    The controller can directly modify self._raw_alpha.data
    """
    def __init__(self, initial_raw_alpha_value: float = 0.0): 
        super().__init__()
        # Parameter for storing the logit of alpha, allows unconstrained updates by controller
        self._raw_alpha = nn.Parameter(torch.tensor([initial_raw_alpha_value], dtype=torch.float32))
        logger.info(f"Initialized WeightedAverage with _raw_alpha starting at: {self._raw_alpha.item():.3f} (alpha ~ {torch.sigmoid(self._raw_alpha).item():.3f})")

    @property
    def alpha(self) -> torch.Tensor:
        """Returns the sigmoid-squashed alpha, ensuring it's between 0 and 1."""
        return torch.sigmoid(self._raw_alpha)

    def forward(
        self,
        p_text: torch.Tensor,  # Shape: [B, 1], probabilities
        p_image: torch.Tensor, # Shape: [B, 1], probabilities
    ) -> torch.Tensor:       # Shape: [B, 1], fused probability
        # Ensure alpha is on the same device as the input probabilities
        current_alpha_val = self.alpha.to(p_text.device) 
        fused_prob = (1 - current_alpha_val) * p_text + current_alpha_val * p_image
        return fused_prob


class Fusion(nn.Module):
    """
    Wrapper for different fusion strategies (e.g., weighted average or a small MLP).
    """
    def __init__(self, config: Dict[str, Any], text_feat_dim: int, image_feat_dim: int):
        super().__init__()
        self.config = config
        self.fusion_type = config.get("fusion_type", "weighted_average")

        if self.fusion_type == "weighted_average":
            initial_raw_alpha = config.get("initial_raw_alpha", 0.0)
            self.impl = WeightedAverage(initial_raw_alpha_value=initial_raw_alpha)
            # Store a direct reference if needed by controller for easy access
            self.weighted_average_submodule = self.impl 
        elif self.fusion_type == "network":
            # Input to network fusion: [p_text, p_image, text_emb, img_emb]
            # Dimensions: 1 (p_text) + 1 (p_image) + text_feat_dim + image_feat_dim
            in_dim = 2 + text_feat_dim + image_feat_dim
            fusion_net_hidden_dim = config.get("fusion_net_hidden_dim", max(16, in_dim // 4)) 
            
            self.impl = nn.Sequential(
                nn.Linear(in_dim, fusion_net_hidden_dim),
                nn.ReLU(),
                nn.Linear(fusion_net_hidden_dim, 1),
            )
            self.network_outputs_logits = True 
            logger.info(f"Initialized 'network' fusion: InDim={in_dim}, HiddenDim={fusion_net_hidden_dim}. Outputs logits.")
        else:
            raise ValueError(f"Unsupported fusion_type '{self.fusion_type}' in Fusion module config.")

    def forward(
        self,
        text_emb: torch.Tensor, text_prob: torch.Tensor,
        img_emb: torch.Tensor,  img_prob: torch.Tensor,
    ) -> torch.Tensor: # Returns probabilities [B,1]
        
        if self.fusion_type == "weighted_average":
            return self.impl(text_prob, img_prob) # WeightedAverage outputs probabilities
        
        elif self.fusion_type == "network":
            # Ensure all inputs are on the same device
            text_prob_d = text_prob.to(text_emb.device)
            img_prob_d = img_prob.to(img_emb.device) # Assuming img_emb is on correct device

            concat_features = torch.cat(
                (text_prob_d, img_prob_d, text_emb, img_emb), dim=-1
            )
            fusion_output = self.impl(concat_features) # Logits from the network
            
            if self.network_outputs_logits:
                return torch.sigmoid(fusion_output) # Convert logits to probabilities
            return fusion_output # If network already outputs probabilities 
        else:
             raise RuntimeError(f"Invalid fusion type '{self.fusion_type}' during forward pass.")


def build_ddp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Wraps a model in DistributedDataParallel if DDP is initialized and device is CUDA.
    Assumes the model is already on the target GPU for the current rank.
    """
    if dist.is_available() and dist.is_initialized() and device.type == 'cuda':
        # When using torchrun, LOCAL_RANK environment variable is set.
        # The device passed should correspond to this local_rank's GPU.
        local_rank_gpu_index = device.index # The GPU index for this DDP process
        
        logger.info(f"Wrapping model {model.__class__.__name__} on rank {dist.get_rank()} (GPU: {local_rank_gpu_index}) in DDP.")
        
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank_gpu_index],    # List of GPU IDs managed by this process
            output_device=local_rank_gpu_index, # Where to gather outputs
            find_unused_parameters=True # Set to True if your model has parameters not used in forward pass under certain conditions
        )
        return ddp_model
    return model # Return original model if DDP is not applicable