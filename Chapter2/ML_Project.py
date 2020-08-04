import os
import tarfile
import urllib.request # 책에서는 import urllib으로 하지만 urllib.request를 import해줘야 Attribute Error가 발생하지 않음.
import pandas as pd

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing") # 경로를 병합하여 새 경로를 생성할 때 os.path.join('경로1', '경로2' ...)을 사용함. 
                                                   # 참고: http://pythonstudy.xyz/python/article/507-%ED%8C%8C%EC%9D%BC%EA%B3%BC-%EB%94%94%EB%A0%89%ED%86%A0%EB%A6%AC
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH): # url을 입력받아 데이터를 path에 다운받는 함수 (데이터를 내려받는 일을 자동화하기 위한 함수)
    os.makedirs(housing_path, exist_ok = True) #housing path에 맞춰 폴더를 생성함. exist_ok=True를 쓴 것은 폴더가 존재하지 않으면 생성하고, 존재하는 경우에는 아무것도 하지 않기 위함.(에러 방지)
    tgz_path = os.path.join(housing_path, "housing.tgz") 
    urllib.request.urlretrieve(housing_url, tgz_path) #urllib.request.urlretrieve를 통해 housing_url의 파일을 tgz_path로 저장할 수 있음.
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

fetch_housing_data(HOUSING_URL, HOUSING_PATH)
load_housing_data(HOUSING_PATH)


