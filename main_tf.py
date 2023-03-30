import cv2
import tensorflow as tf
from scipy.spatial import distance


def keep_ratio_resize(img, width=256):
    aspect_ratio = float(width) / img.shape[1]
    dsize = (width, int(img.shape[0] * aspect_ratio))
    resized = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)

    return resized


def expand_dimension(img, axis=0):
    return tf.expand_dims(img, axis=axis)


def get_feature_vector(img):
    VGG19_Feature_Extractor = tf.keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling='max')

    return VGG19_Feature_Extractor(img)


def calculate_similarity(feature_vector1, feature_vector2):
    return 1 - distance.cosine(tf.squeeze(feature_vector1), tf.squeeze(feature_vector2))


def read_image(image: str):
    return cv2.imread(image)


if __name__ == '__main__':
    img1 = expand_dimension(keep_ratio_resize(read_image("./15.jpg")))
    img2 = expand_dimension(keep_ratio_resize(read_image("./desktop_artwork_3.jpg")))
    img3 = expand_dimension(keep_ratio_resize(read_image("./desktop_artwork_19.jpg")))

    f1 = get_feature_vector(img1)
    f2 = get_feature_vector(img2)
    f3 = get_feature_vector(img3)
    print(calculate_similarity(f1, f2))
    print(calculate_similarity(f1, f3))
    print(calculate_similarity(f2, f3))
