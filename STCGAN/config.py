import os
from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class TrainingConfig:
    """
    Configuration for STCGAN training
    """
    # Experiment and Logging
    experiment_name: str = "STCGAN_Shadow_Removal"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    
    # Dataset Configuration
    dataset_path: str = "../ISTD_DATASET"
    train_subset: str = "train"
    test_subset: str = "test"
    
    # Model Hyperparameters
    generator_base_filters: int = 64
    discriminator_base_filters: int = 64
    latent_dim: int = 100
    
    # Training Hyperparameters
    batch_size: int = 16
    epochs: int = 200
    learning_rate: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    
    # Loss Weights
    data_loss_weight: float = 100.0
    adversarial_loss_weight: float = 1.0
    perceptual_loss_weight: float = 10.0
    
    # Augmentation
    random_crop_size: int = 256
    horizontal_flip_prob: float = 0.5
    random_rotation_angle: float = 10.0
    
    # Hardware and Performance
    use_mixed_precision: bool = True
    use_xla: bool = True
    gpu_memory_limit: Optional[int] = None
    
    # Validation and Monitoring
    validation_frequency: int = 5
    save_checkpoint_frequency: int = 10
    max_to_keep: int = 5
    
    # Inference
    inference_batch_size: int = 1
    
    def __post_init__(self):
        """
        Post-initialization setup and validation
        """
        # Create necessary directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Validate hyperparameters
        assert self.batch_size > 0, "Batch size must be positive"
        assert 0 < self.learning_rate < 1, "Learning rate must be between 0 and 1"


@dataclass
class AugmentationConfig:
    """
    Detailed configuration for data augmentation
    """
    random_crop: bool = True
    crop_size: int = 256
    
    horizontal_flip: bool = True
    flip_probability: float = 0.5
    
    random_rotation: bool = True
    max_rotation_angle: float = 10.0
    
    random_scale: bool = True
    scale_factor: float = 0.1
    
    color_jitter: bool = False
    brightness_delta: float = 0.1
    contrast_delta: float = 0.1
    saturation_delta: float = 0.1
    hue_delta: float = 0.1


@dataclass
class InferenceConfig:
    """
    Configuration for model inference
    """
    model_checkpoint: str = "./checkpoints/best_model"
    input_image_dir: str = "./test_images"
    output_image_dir: str = "./output_images"
    
    # Inference-specific parameters
    use_tta: bool = False  # Test Time Augmentation
    tta_methods: List[str] = field(default_factory=lambda: ['flip', 'rotate'])
    
    # Post-processing
    apply_color_correction: bool = True
    normalize_output: bool = True


def get_config(mode: str = 'train') -> Union[TrainingConfig, InferenceConfig]:
    """
    Retrieve configuration based on mode
    
    Args:
        mode (str): Configuration mode ('train' or 'inference')
    
    Returns:
        Configuration object
    """
    if mode == 'train':
        return TrainingConfig()
    elif mode == 'inference':
        return InferenceConfig()
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'train' or 'inference'.")


# Augmentation configuration
augmentation_config = AugmentationConfig()

# Export configurations for easy access
__all__ = [
    'TrainingConfig', 
    'InferenceConfig', 
    'AugmentationConfig', 
    'get_config', 
    'augmentation_config'
]
