"""
데이터에서 테스트 세트를 분리하는 과정임.

테스트 세트는 모델을 훈련시키는데 사용되는 훈련세트와 달리 모델의 오차를 평가하는데 사용됨.

테스트 세트로 일반화 오차를 추정할 경우 데이터 스누핑 편향이 발생할 수 있기에 이번 단계에서는 단순히 전체 데이터에서 일부를 테스트 세트로 분리하는 정도만 진행함.

데이터 셋의 20% 정도를 무작위로 추출해서 떼어 놓으면 테스트 세트를 만들 수 있음. 다만 전체 데이터 셋이 매우 큰 경우에는 적정 선에서 테스트 세트를 분리하면 됨.


"""

#1. 프로그램 재실행을 염두하지 않고 단순히 데이터 셋에서 테스트 세트를 분리하는 방법.
from ML_Project import load_housing_data 
import pandas as pd
import numpy as np

housing = load_housing_data()

def split_train_test(data, test_ratio): 
    shuffled_indices = np.random.permutation(len(data)) #np.random.permutation(숫자) : 0 ~ (주어진 숫자 - 1)까지의 숫자들을 무작위로 섞인 배열을 만듦.
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size] # 이경우에 test_indices는 [3124 2145 324 ... ] 같은 배열을 반환함.
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices] # iloc은 행을 선택하는데 사용함. test_indices, train_indices가 랜덤한 숫자를 갖는 배열이여서
                                                             # train_set과 test_set이 랜덤하게 선택되는 것임.
                                                             # 참고 : https://azanewta.tistory.com/34

train_set, test_set = split_train_test(housing, 0.2)
