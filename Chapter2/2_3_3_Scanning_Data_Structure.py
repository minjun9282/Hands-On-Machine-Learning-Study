"""
데이터에서 테스트 세트를 분리하기 전에 대략적으로 데이터의 구조를 훑어보는 과정임.

데이터에 샘플 갯수는 몇개 인지, 특성은 몇가지가 이고 특성의 타입은 무엇인지, 값들의 분포는 어떻게 되는지(히스토그램) 등을 간략하게 파악하는 단계임.

데이터를 훑어보며 특성들의 스케일을 어떻게 맞춰 줄 것인지, NULL TYPE의 데이터는 어떻게 처리할 것인지, 데이터를 종모양으로 변형시켜야 하는지도 생각해 봐야함.
"""

from ML_Project import load_housing_data #2.3.3 데이터 구조 훑어보기를 앞의 ML_Project(데이터 내려받고 pandas로 읽어 들임) 파일과 분리함.
import pandas as pd
import matplotlib.pyplot as plt

housing = load_housing_data()
print(housing.head()) #jupyter notebook과 다르게 print()함수를 사용해줘야 화면에 출력 가능함.

"""
housing.head()의 출력 결과 : 각 행은 하나의 구역을 나타내며 특성은 10가지인 것을 파악할 수 있다.

   longitude  latitude  housing_median_age  total_rooms  total_bedrooms  population  households  median_income  median_house_value ocean_proximity
0    -122.23     37.88                41.0        880.0           129.0       322.0       126.0         8.3252            452600.0        NEAR BAY
1    -122.22     37.86                21.0       7099.0          1106.0      2401.0      1138.0         8.3014            358500.0        NEAR BAY
2    -122.24     37.85                52.0       1467.0           190.0       496.0       177.0         7.2574            352100.0        NEAR BAY
3    -122.25     37.85                52.0       1274.0           235.0       558.0       219.0         5.6431            341300.0        NEAR BAY
4    -122.25     37.85                52.0       1627.0           280.0       565.0       259.0         3.8462            342200.0        NEAR BAY

"""

print(housing.info()) # info() : 데이터의 전체 행수, 각 특성의 데이터 타입과 NULL이 아닌 값의 개수를 확인하는데 유용함.

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20640 entries, 0 to 20639
Data columns (total 10 columns):
 #   Column              Non-Null Count  Dtype
---  ------              --------------  -----
 0   longitude           20640 non-null  float64
 1   latitude            20640 non-null  float64
 2   housing_median_age  20640 non-null  float64
 3   total_rooms         20640 non-null  float64
 4   total_bedrooms      20433 non-null  float64 -> 207개 구역은 total_bedrooms의 값이 NULL인것을 알 수 있음.
 5   population          20640 non-null  float64
 6   households          20640 non-null  float64
 7   median_income       20640 non-null  float64
 8   median_house_value  20640 non-null  float64
 9   ocean_proximity     20640 non-null  object -> 다른 특성과 다른 데이터 타입(CSV파일을 읽었을 경우에는 주로 text가 object type으로 읽힘)
dtypes: float64(9), object(1)
memory usage: 1.5+ MB
None
"""

print(housing["ocean_proximity"].value_counts())
"""
ocean_proximity 특성이 categorical한 특성이기에 어떤 카테고리가 있으며 각 카테고리에 속하는 구역의 수를 확인하기 위해서 housing["특성 이름"].value_counts()를 사용함.

<1H OCEAN     9136
INLAND        6551
NEAR OCEAN    2658
NEAR BAY      2290
ISLAND           5
"""

print(housing.describe())
"""
Name: ocean_proximity, dtype: int64 -> 숫자형이 아닌 특성을 따로 표시함.
          longitude      latitude  housing_median_age   total_rooms  total_bedrooms    population    households  median_income  median_house_value
count  20640.000000  20640.000000        20640.000000  20640.000000    20433.000000  20640.000000  20640.000000   20640.000000        20640.000000
mean    -119.569704     35.631861           28.639486   2635.763081      537.870553   1425.476744    499.539680       3.870671       206855.816909
std        2.003532      2.135952           12.585558   2181.615252      421.385070   1132.462122    382.329753       1.899822       115395.615874
min     -124.350000     32.540000            1.000000      2.000000        1.000000      3.000000      1.000000       0.499900        14999.000000
25%     -121.800000     33.930000           18.000000   1447.750000      296.000000    787.000000    280.000000       2.563400       119600.000000
50%     -118.490000     34.260000           29.000000   2127.000000      435.000000   1166.000000    409.000000       3.534800       179700.000000
75%     -118.010000     37.710000           37.000000   3148.000000      647.000000   1725.000000    605.000000       4.743250       264725.000000
max     -114.310000     41.950000           52.000000  39320.000000     6445.000000  35682.000000   6082.000000      15.000100       500001.000000

=> 숫자형 특성의 요약 정보를 보여줌. (각 특성의 갯수, 표준편차, 중간 값, 최대값, 최소값 등등)
"""

housing.hist(bins=50, figsize = (20, 15))
plt.show()