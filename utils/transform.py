from torchvision import transforms, datasets


get_transform = lambda resize, crop=None: transforms.Compose(
    [
        transforms.Resize(resize),
        transforms.CenterCrop(resize if crop is None else crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Imagenet standards
    ]
)
