from torchvision import transforms
transform_image224_train = transforms.Compose([

    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_image224_test = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
 transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# using mean and std in recon transform cannot deal with raw image val and test
transform_image224_4Tensor = transforms.Compose([

    # transforms.RandomCrop(224, padding=4),
    # in DRR version, it has scale (0.7, 1.0); in IB, this improves the performance by 8%
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),

])

# feature_mean = [0.4903, 0.4881, 0.4975]
# feature_std = [0.2944, 0.2908, 0.2787]
feature_mean = [-0.0154, -0.0335,  0.0038]
feature_std = [1.1844, 1.1839, 1.2170]
transform_image224_4Tensor_pil_train = transforms.Compose([

    # transforms.RandomCrop(224, padding=4),
    # in DRR version, it has scale (0.7, 1.0); in IB, this improves the performance by 8%
    # transforms.ToPILImage(mode='RGB'),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(mean=feature_mean, std=feature_std),
    # transforms.ToTensor(),

])
transform_image224_4Tensor_pil_test = transforms.Compose([

    # transforms.RandomCrop(224, padding=4),
    # in DRR version, it has scale (0.7, 1.0); in IB, this improves the performance by 8%
    # transforms.ToPILImage(mode='RGB'),
    transforms.Normalize(mean=feature_mean, std=feature_std),
    # transforms.ToTensor(),

])

transform_recon_cifar_4Tensor = transforms.Compose([

    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),

])