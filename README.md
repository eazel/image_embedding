# image_embedding

**main_torch.py**: eazel image (dashboard.artwork)가 주어졌을때 해당 이미지를 벡터값으로 변환 후, 나중에 이미지끼리의 유사도를 계산하기 위해 dashboard.artwork_vector 테이블에 이미지의 벡터값을 미리 저장하는 하는 코드입니다.  
**top10_images.py**: 미리 저장한 이미지들의 벡터값 (dashboard.artwork_vector) 을 읽어서 원하는 target 이미지와 유사한 top 10 이미지를 로드하는 코드입니다. 
