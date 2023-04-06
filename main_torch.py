import cv2
import torch
import torchvision.models as models
import eazel.image
from scipy.spatial import distance
from torch.nn import AvgPool2d
from dotenv import load_dotenv
from eazel.db import PostgresqlManager
import psycopg2
import os
import retrying

dotenv_path = ".env"
load_dotenv(dotenv_path)
dbm = PostgresqlManager()


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


@retrying.retry(stop_max_attempt_number=5, wait_fixed=1000)
def get_artworks_list():
    sql = f"""
        select id, file_name from dashboard.artwork
    """

    try:
        artwork_list = dbm.execute_sql(sql, fetchall=True)
        return artwork_list
    except (Exception, psycopg2.DatabaseError) as e:
        print(str(e))


@retrying.retry(stop_max_attempt_number=5, wait_fixed=1000)
def insert_to_artwork_vector(artwork_id, embedded_vector, file_name):
    if embedded_vector is None:
        inserting_vector = None
    else:
        embedded_vector_bytes = embedded_vector.numpy().tobytes()
        inserting_vector = psycopg2.Binary(embedded_vector_bytes)

    sql = f"""
        insert into dashboard.artwork_vector (
            artwork_id,
            embedded_vector,
            file_name
        )
        values (%s,%s,%s)
        """
    try:
        dbm.execute_sql(sql,
                        parameters=(
                            artwork_id,
                            inserting_vector,
                            file_name
                        ),
                        commit=True)
    except (Exception, psycopg2.DatabaseError) as e:
        print(str(e))


if __name__ == '__main__':
    artworks = get_artworks_list()[9316:]

    for idx, artwork in enumerate(artworks):
        artwork_id = artwork[0]
        file_name = artwork[1]

        print(f"idx: {idx}: ", artwork_id, file_name)

        downloaded_image = eazel.image.download_eazel_artwork(artwork_id=artwork_id,
                                                              file_name=file_name,
                                                              folder_name='s3_images')
        if downloaded_image is None or 'png' in file_name or 'gif' in file_name:
            feature_vector = None
            insert_to_artwork_vector(artwork_id, feature_vector, file_name)
            os.remove(f'./s3_images/{file_name}')
            continue

        img = keep_ratio_resize(read_image(f"./s3_images/{file_name}"))
        feature_vector = get_feature_vector_vgg19(img)
        feature_vector = get_flattened_vector(feature_vector)

        insert_to_artwork_vector(artwork_id, feature_vector, file_name)
        os.remove(f'./s3_images/{file_name}')
