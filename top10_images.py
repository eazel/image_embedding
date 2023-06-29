'''
dashboard.artwork_vector 테이블에서 벡터를 읽어와서 원하는 이미지를 target으로 했을때, 
해당 이미지와 비슷하다고 (1-consine distance >= 0.88 인것) 생각되는 이미지 10개를 가져오는 코드. 
'''
import numpy as np
from dotenv import load_dotenv
from eazel.db import PostgresqlManager
import retrying
import psycopg2
from scipy.spatial import distance
import eazel.image
from matplotlib import pyplot as plt
import torch

dotenv_path = ".env"
load_dotenv(dotenv_path)
dbm = PostgresqlManager()


@retrying.retry(stop_max_attempt_number=5, wait_fixed=1000)
def read_vector_table():
    sql = f"""
        select * from dashboard.artwork_vector 
    """

    try:
        result = dbm.execute_sql(sql, fetchall=True)
        return result
    except (Exception, psycopg2.DatabaseError) as e:
        print(str(e))


def calculate_similarity(feature_vector1, feature_vector2):
    return 1 - distance.cosine(feature_vector1, feature_vector2)


def get_top10_images(input_img, image_info):
    input_id, input_vector, input_file_name = input_img

    input_tensor = torch.from_numpy(np.frombuffer(input_vector, dtype=np.float32))
    similar_images = []

    for i, image in enumerate(image_info):
        image_id, image_vector, file_name = image
        if image_vector is None:
            continue
        image_tensor = torch.from_numpy(np.frombuffer(image_vector, dtype=np.float32))
        if image_id == input_id:
            continue
        else:
            similarity = calculate_similarity(input_tensor, image_tensor)
            if similarity >= 0.88 and len(similar_images) < 10:
                similar_images.append((image_id, similarity, file_name))
            if len(similar_images) == 10:
                break
    return similar_images


def show_similar_images(similar_images):
    similar_images.sort(key=lambda x: x[1], reverse=True)

    for image in similar_images:
        image_id, similarity, file_name = image
        print(f"Similarity: {similarity}")
        print(f"Image id: {image_id}")
        print(f"Image name: {file_name}")
        print("======================================")
        eazel.image.download_eazel_artwork(artwork_id=image_id,
                                           file_name=file_name,
                                           folder_name='similar_images')

        plt.figure()
        plt.imshow(plt.imread(f"similar_images/{file_name}"))
        plt.show()


if __name__ == '__main__':
    image_info = read_vector_table()
    target_image = image_info[1776]
    if target_image[1] is not None:
        similar_images = get_top10_images(target_image, image_info)

        image_id, similarity, file_name = target_image
        print(f"original_image: {file_name}")

        if len(similar_images) > 0:
            eazel.image.download_eazel_artwork(artwork_id=image_id,
                                               file_name=file_name,
                                               folder_name='similar_images')
            plt.figure()
            plt.imshow(plt.imread(f"similar_images/{file_name}"))
            plt.show()
            show_similar_images(similar_images)
