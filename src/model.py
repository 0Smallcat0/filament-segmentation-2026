"""Semantic U-Net model used by train and inference."""
import segmentation_models_pytorch as smp


def build_model(pretrained=False):
    """Build a small ResNet-18 encoder U-Net; loading weights needs no internet."""
    return smp.Unet(encoder_name='resnet18', encoder_weights='imagenet' if pretrained else None,
                    in_channels=1, classes=1)
