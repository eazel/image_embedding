import cv2
import torch
import torchvision.models as models
from scipy.spatial import distance
from torch.nn import AvgPool2d


def keep_ratio_resize(img, width=256):
    aspect_ratio = float(width) / img.shape[1]
    dsize = (width, int(img.shape[0] * aspect_ratio))
    resized = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)

    return resized


def get_feature_vector_vgg19(img):
    vgg19 = models.vgg19(pretrained=True)
    vgg19 = vgg19.features
    vgg19 = torch.nn.Sequential(*list(vgg19.children()))
    vgg19.eval()

    img = torch.unsqueeze(torch.from_numpy(img), 0).permute(0, 3, 1, 2)

    with torch.no_grad():
        feature_vector = vgg19(img.float() / 255.0)

    return feature_vector


def calculate_similarity(feature_vector1, feature_vector2):
    return 1 - distance.cosine(feature_vector1, feature_vector2)


def get_flattened_vector(feature_vector):
    vector_shape = feature_vector.shape[2:]

    vector_pooling_layer = AvgPool2d(vector_shape)

    pooled_vector = vector_pooling_layer(feature_vector)

    return torch.squeeze(pooled_vector)


def read_image(image: str):
    return cv2.imread(image)


if __name__ == '__main__':
    img1 = keep_ratio_resize(read_image("./sample_images/00095_f.jpg"))
    img2 = keep_ratio_resize(read_image("./sample_images/00101_f.jpg"))
    img3 = keep_ratio_resize(read_image("./sample_images/00102_f.jpg"))
    img4 = keep_ratio_resize(read_image("./sample_images/giraffe.jpg"))
    img5 = keep_ratio_resize(read_image("./sample_images/ocean1.jpg"))
    img6 = keep_ratio_resize(read_image("./sample_images/ocean2.jpg"))

    f1 = get_feature_vector_vgg19(img1)
    f2 = get_feature_vector_vgg19(img2)
    f3 = get_feature_vector_vgg19(img3)
    f4 = get_feature_vector_vgg19(img4)
    f5 = get_feature_vector_vgg19(img5)
    f6 = get_feature_vector_vgg19(img6)

    f1 = get_flattened_vector(f1)
    f2 = get_flattened_vector(f2)
    f3 = get_flattened_vector(f3)
    f4 = get_flattened_vector(f4)
    f5 = get_flattened_vector(f5)
    f6 = get_flattened_vector(f6)

    print(f6.dtype)
    print(calculate_similarity(f1, f2))
    print(calculate_similarity(f1, f3))
    print(calculate_similarity(f2, f3))

    print(calculate_similarity(f4, f5))
    print(calculate_similarity(f4, f6))
    print(calculate_similarity(f5, f6))
