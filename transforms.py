import torchvision.transforms as T


PLANTNET_MEAN = (0.4425, 0.4695, 0.3266)
PLANTNET_STD = (0.2353, 0.2219, 0.2325)


class TrainTransform:
    def __init__(self, crop_size=224, scale_min=0.4):
        self.transform = T.Compose(
            [
                T.RandomResizedCrop(crop_size, scale=(scale_min, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.ToTensor(),
                T.Normalize(mean=PLANTNET_MEAN, std=PLANTNET_STD),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class ValTransform:
    def __init__(self, resize_size=224, crop_size=224):
        self.transform = T.Compose(
            [
                T.Resize(
                    (resize_size, resize_size),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.CenterCrop(crop_size),
                T.ToTensor(),
                T.Normalize(mean=PLANTNET_MEAN, std=PLANTNET_STD),
            ]
        )

    def __call__(self, image):
        return self.transform(image)
