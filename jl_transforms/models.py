from torch import nn
from torchvision import models  # type: ignore[import]


def get_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    model = models.resnet18(weights=None)

    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=old_conv.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=old_conv.bias is not None,
    )

    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
