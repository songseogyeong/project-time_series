# SM 엔터테인먼트 주가 예측

<br>

<img src='./images/smlog.jpg' width='800px'>

<br>

2024.06.17.

<br></br>
<br></br>
<br></br>
<br></br>
<br></br>

# Ⅰ. 프로젝트 개요
## 1. 프로젝트 목표
SM 엔터테인먼트 주가를 분석하여 수익 및 변동성 등의 현황을 파악하고, 최적의 모델을 통해 미래 예측 및 투자 전략을 발굴한다

<br></br>
<br></br>

## 2. 프로젝트 기대효과
✔ 미래 주가 예측
과거 데이터를 바탕으로 주가의 향후 움직임을 예측하여 투자 전략을 수립할 수 있음

<br>

✔ 투자 리스크 평가 및 관리
변동성을 분석하여 투자 리스크를 평가하고 관리할 수 있음

<br></br>
<br></br>

## 3. 데이터 흐름
### ○ 데이터 분석 프로세스

<br>

✔ 데이터 분석

<img src='./images/01_개요/데이터분석흐름.png' width='800px'>

<br>

✔ 머신러닝

<img src='./images/01_개요/머신러닝흐름.png' width='800px'>

<br>

✔ 딥러닝

<img src='./images/01_개요/딥러닝흐름.png' width='800px'>

<br></br>
<br></br>

## 4. 데이터 수집
✔ 데이터 정보  
Yahoo Finance의 Yfinance 라이브러리를 통한 금융 데이터 활용

<br>

✔ SM 엔터테인먼트
- 코스닥(KOSDAQ) 시장에서 거래
- SM엔터테인먼트는 대한민국의 대표적인 엔터테인먼트 기업
- 다양한 엔터테인먼트 사업을 통해 수익을 창출

<br>

✔ KODEX 골드선물(H)
- KRX (한국거래소)에서 거래
- 상장수지펀드(ETF)로 금의 가격 움직임을 추적하도록 설계
- 투자자들이 실제로 물리적인 금을 소유하지 않고도 금의 가격에 투자할 수 있는 방법을 제공

<br>

✔ 데이터 추출 (2010년 10월 1일 ~ 2024년 6월 17일)

| SM 주가                          | GOLD 주가              |
|:----------------------------:|:------------------:|
| <img src='./images/01_개요/sm주가.png' width='300px'>  | <img src='./images/01_개요/gold주가.png' width='300px'>  |

<details>
  <summary>code</summary>

  ```
  import yfinance as yf

  # SM 엔터테인먼트와 Gold(금 펀드, ETF) 티커를 저장
  columns = ['041510.KQ', '132030.KS']  

  # yfinance 라이브러리를 사용하여 특정 종목 데이터를 다운로드 후 시계열 데이터 프레임으로 변환 (소수점 4자리까지 표기)
  f_df = yf.download(columns, start='2010-10-01', end='2024-06-18')['Adj Close'].round(4)
  f_df
  ```
</details>

<br>

- SM의 주가는 2010년을 기점으로 큰 변동이 발생하고, GOLD 주가는 2010년 10월 1일 부터 시작하므로 해당 일자를 기준으로 현재 시점(2024년 6월 17일)까지 추출

<br></br>
<br></br>
<br></br>
<br></br>
<br></br>

# Ⅱ. 데이터 분석
## 1. 데이터 분석
### ○ info

| Column | Non-Null Count | Dtype   |
|:------:|:--------------:|:-------:|
| SM     | 3369 non-null  | float64 |
| GOLD   | 3350 non-null  | float64 |

<details>
  <summary>code</summary>

  ```
  # 데이터 정보 출력
  pre_f_df.info()
  ```
</details>

<br></br>

### ○ 주가
✔ 2010년 ~ 현재시점까지의 데이터를 보았을 때 SM과 GOLD는 2011년을 기점으로 금액이 상승하였다가 2013년 쯤 하락하였으며,  
  2020년 ~ 2021년을 기점으로 다시 금액이 상승하는 것을 볼 수 있다.  

✔ 주가 변동 패턴이 어느정도 유사하게 나타남에 따라 두 종목 간의 상관관계를 추정해볼 수 있다.

<img src='./images/주가.png' width='800'>
<details>
  <summary>code</summary>

  ```
  import matplotlib.pyplot as plt

  # 그래프 크기 설정
  fig, ax = plt.subplots(2, 1, figsize=(15, 7))

  # 주가 시각화
  pre_f_df.SM.plot(color=colors[0], ax=ax[0])
  ax[0].set_title('SM')
  ax[0].legend()

  pre_f_df.GOLD.plot(color=colors[1], ax=ax[1])
  ax[1].set_title('GOLD')
  ax[1].legend()

  # 레이아웃 조정
  plt.tight_layout()

  # 그래프 표시
  plt.show()
  ```
</details>

<br></br>

### ○ 차분 후 주가
✔ 차분 후 데이터를 확인하였을 때, SM, GOLD 수익률의 변동이 전반적으로 크지 않기 때문에 일반적으로 안정적이라 판단되나  
  특정연도에 수익률이 크게 변동한 것으로 나타나 안정성이 떨어진 것으로 보여진다.

<img src='./images/차분주가.png' width='800'>
<details>
  <summary>code</summary>

  ```
  import matplotlib.pyplot as plt

  # 그래프 크기 설정
  fig, ax = plt.subplots(2, 1, figsize=(15, 7))

  # 차분 후 주가 시각화
  pre_f_df.SM.diff().plot(color=colors[0], ax=ax[0])
  ax[0].set_title('SM')
  ax[0].legend()

  pre_f_df.GOLD.diff().plot(color=colors[1], ax=ax[1])
  ax[1].set_title('GOLD')
  ax[1].legend()

  # 레이아웃 조정
  plt.tight_layout()

  # 그래프 표시
  plt.show()
  ```
</details>

<br></br>

### ○ 변화량 및 변동량
✔ SM과 GOLD의 변화량 및 변동량을 비교한 결과, SM의 변화량과 변동량이 상대적으로 높게 나타났다.  
✔ 2023년 SM의 수익률 변동이 크게 발생한 점을 고려할 때, 단기적인 변동이 장기적인 안정성을 올바르게 반영하지 못할 가능성이 있다.

<img src='./images/변화량변.png' width='800'>
<details>
  <summary>code</summary>

  ```
  import matplotlib.pyplot as plt

  # 그래프 크기 설정
  fig, ax = plt.subplots(1, 2, figsize=(12, 5))

  # 변화율 시각화
  pre_f_df.pct_change().mean().plot(kind='bar', color=colors, edgecolor='black', ax=ax[0])
  ax[0].set_title('변화량')

  # 변동률 시각화
  pre_f_df.pct_change().std().plot(kind='bar', color=colors, edgecolor='black', ax=ax[1])
  ax[1].set_title('변동량')

  # 레이아웃 조정
  plt.tight_layout()

  # 그래프 표시
  plt.show()
  ```
</details>

<br></br>

### ○ 변화량 및 변동량
✔ SM과 GOLD의 변화량 및 변동량을 비교한 결과, SM의 변화량과 변동량이 상대적으로 높게 나타났다.  
✔ 2023년 SM의 수익률 변동이 크게 발생한 점을 고려할 때, 단기적인 변동이 장기적인 안정성을 올바르게 반영하지 못할 가능성이 있다.

<img src='./images/변화량변.png' width='800'>
<details>
  <summary>code</summary>

  ```
  import matplotlib.pyplot as plt

  # 그래프 크기 설정
  fig, ax = plt.subplots(1, 2, figsize=(12, 5))

  # 변화율 시각화
  pre_f_df.pct_change().mean().plot(kind='bar', color=colors, edgecolor='black', ax=ax[0])
  ax[0].set_title('변화량')

  # 변동률 시각화
  pre_f_df.pct_change().std().plot(kind='bar', color=colors, edgecolor='black', ax=ax[1])
  ax[1].set_title('변동량')

  # 레이아웃 조정
  plt.tight_layout()

  # 그래프 표시
  plt.show()
  ```
</details>

<br></br>

### ○ 수익률 및 변동률
✔ 현재 시점에서 과거의 시점을 기준으로 SM과 GOLD의 수익률을 계산하고 시각화했다.  
✔ SM의 수익률이 더 큰 것을 확인할 수 있었지만, 변동률은 GOLD가 더 안정적으로 나타났다.  
✔ 고수익을 원하는 투자자는 SM을 선호할 수 있지만, 안정적인 투자를 원하는 경우에는 GLD를 선택할 수 있다.

<img src='./images/수익률변동률.png' width='800'>
<details>
  <summary>수익률 code</summary>

  ```
  import numpy as np

  # 수익률 계산
  # 각 날짜의 주가를 전날 주가로 나눈 후 로그를 취해 수익률 계산
  # 로그를 취해 백분율로 변환
  rate_f_df = np.log(pre_f_df / pre_f_df.shift(1))
  rate_f_df
  ```
</details>
<details>
  <summary>변동률 code</summary>

  ```
  # 변동률 계산하기
  # 변동률 = (오늘 종가 - 어제 종가) / 어제 종가
  pct_f_df = pre_f_df.pct_change()
  pct_f_df
  ```
</details>
<details>
  <summary>그래프 code</summary>

  ```
  import matplotlib.pyplot as plt

  # 그래프 크기 설정
  fig, ax = plt.subplots(2, 1, figsize=(15, 8))

  # 특정 데이터 수익률 시각화
  # lw = line widht = 선 두께
  rate_f_df[['SM']].plot(color=colors[0], lw=0.5, ax=ax[0])
  rate_f_df[['GOLD']].plot(color='blue', lw=0.5, ax=ax[0], alpha=0.6)
  ax[0].set_title('수익률')

  # 특정 데이터 수익률 시각화
  # lw = line widht = 선 두께
  pct_f_df[['SM']].plot(color=colors, lw=0.5, ax=ax[1])
  pct_f_df[['GOLD']].plot(color='blue', lw=0.5, ax=ax[1], alpha=0.6)
  ax[1].set_title('변동률')

  # 레이아웃 조정
  plt.tight_layout()

  # 그래프 표시
  plt.show()
  ```
</details>

<br></br>

### ○ 일간 수익률

<img src='./images/일간수익.png' width='800'>
<details>
  <summary>code</summary>

  ```
  import matplotlib.pyplot as plt

  # 일간 수익률
  # cumsum = 각 원소들의 누적합의 과정
  # exp: 지수 (로그 후 지수 하면 원래 값을 얻을 수 있음)
  rate_f_df.cumsum().apply(np.exp).plot(figsize=(15, 4), color=colors)
  plt.show()
  ```
</details>

<br></br>

### ○ 월간 수익률

<img src='./images/월간수익.png' width='800'>
<details>
  <summary>code</summary>

  ```
  # 월간 수익률
  # resample: 단위 리샘플링, 1m = 월 단위로 리샘플링
  # last: 리샘플링된 그룹에서 가장 마지막 값을 선택 (월간은 가장 마지막 값이 필요)
  # 즉, 각 월의 마지막 날에 해당하는 데이터 포인트를 선택
  rate_f_df.cumsum().apply(np.exp).resample('1m').last().plot(figsize=(15, 4), color=colors)
  plt.show()
  ```
</details>

<br></br>

### ○ 분석
✔ SM의 일간 수익률과 연간 수익률은 2011년 말과 2022년에 변동성이 크게 나타났지만, GOLD는 안정적인 패턴을 보였다.  
✔ 월간 수익률은 각 월의 마지막 값을 기준으로 측정되었으며, 월간 수익률 그래프는 일간 수익률 그래프보다 변동성이 적고 더 안정적이다.

<br></br>

**✔ 2011년**

<img src='./images/sm06.png' width=500px>

<br>

- 2011년 하반기: 그룹 EXO의 데뷔로 인해 수익률이 크게 상승했다.

<br></br>

**✔ 2021년**

<div>
    <img src='./images/sm07.png' width=500px>
    <img src='./images/sm08.png' width=500px>
</div>

<br>

- 5월: 그룹 NCT DREAM의 활동으로 인해 수익률이 크게 상승했다.

- 9월: 그룹 NCT 127의 선주문량 및 판매량이 최다 기록을 세우며 수익률 변동성이 크게 나타났다.

<br></br>

**✔ 2023년**

<div>
    <img src='./images/sm10.png' width=500px>
    <img src='./images/sm09.png' width=500px>
</div>

<br>

- 2023년 상반기: 그룹 NCT 127의 음반 판매 및 월드투어 개최 성공으로 수익률이 크게 성장했다.

<br>

<img src='./images/sm12.png' width=500px>

<br>

- 2023년 하반기: 그룹 라이즈의 멤버 홍승한 군의 사생활 논란으로 인하여 수익률이 크게 하락했다.

<br></br>

**✔ 2024년**

<img src='./images/sm11.png' width=500px>

<br>

- 2024년 상반기: 그룹 에스파의 멤버 카리나 양의 연애설로 인해 수익률이 크게 하락했다.

<br></br>

### ○ 연간 연율화
✔ 연간 영업일을 약 252일로 두고 연간 연율화를 계산했을 때, SM이 약 -0.2286, GOLD가 0.1051으로 나타났다.  
✔ 최근 1년 간 GOLD가 긍정적인 성과를 보이며 안정적인 투자 대안임을 나타내고, SM은 부정적인 성과를 보여 주의가 필요하다.

| Column | 연율화     |
|:------:|:-------:|
| SM     | -0.2286 |
| GOLD   | 0.1051  |

<details>
  <summary>code</summary>

  ```
  # 연간 연율화

  # 연율화 계산을 위해 최근 1년 데이터 추출
  pre_rate_f_df = rate_f_df[-252:]

  # 연간 영업일(약 252일로 계산)
  annualized_mean = pre_rate_f_df.mean() * 252
  annualized_mean
  ```
</details>

<br></br>

### ○ VIF
✔ SM과 GOLD의 다중공선성을 확인하여 두 종목 간의 상관관계를 평가한 결과, 각 VIF 점수가 약 1로 나타나며 매우 낮은 상관관계를 보였다.  
✔ 따라서 주가 변동 패턴이 유사하다고 해서 두 종목 간에 상관관계가 있다고 단정할 수 없었고, 각 종목은 서로 독립적이며 선형 관계가 거의 없음을 확인했다.

| Column | 연율화     |
|:------:|:-------:|
| SM     | 1.0001 |
| GOLD   | 1.0001  |

<details>
  <summary>함수 선언 code</summary>

  ```
  import pandas as pd
  from statsmodels.stats.outliers_influence import variance_inflation_factor

  # 다중 공산성 평가 지표 VIF 함수 선언
  def get_vif(features):
      vif = pd.DataFrame()
      vif['vif_score'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
      vif['feature'] = features.columns
      return vif
  ```
</details>
<details>
  <summary>VIF 확인 code</summary>

  ```
  # NaN 값 제거 후 VIF 확인
  rate_f_df = rate_f_df.dropna()
  get_vif(rate_f_df)
  ```
</details>

<br></br>

### ○ 분포 및 log 변환
✔ 데이터 분포와 로그 변환된 데이터 분포를 비교하엿을 때, 정규분포에 가까워진 것을 확인할 수 있다.  
✔ 이로 인해 신뢰성이 향상되었으며, 로그 변환을 통해 일정한 척도로 비교할 수 있는 편의성이 증가했다. 

<img src='./images/분포변환.png' width='800'>
<details>
  <summary>code</summary>

  ```
  import matplotlib.pyplot as plt
  import numpy as np

  # 그래프 크기 설정
  fig, ax = plt.subplots(2, 2, figsize=(15, 9))

  # SM 분포 시각화
  pre_f_df.SM.hist(color=colors[0], edgecolor='black', bins=50, ax=ax[0][0])
  ax[0][0].set_title('SM 분포')

  # GLD 분포 시각화
  pre_f_df.GOLD.hist(color=colors[1], edgecolor='black', bins=50, ax=ax[0][1])
  ax[0][1].set_title('GOLD 분포')

  # SM 분포 시각화
  rate_f_df.SM.hist(color=colors[0], edgecolor='black', bins=50, ax=ax[1][0])
  ax[1][0].set_title('SM 분포 log 변환')

  # GLD 분포 시각화
  rate_f_df.GOLD.hist(color=colors[1], edgecolor='black', bins=50, ax=ax[1][1])
  ax[1][1].set_title('GOLD 분포 log 변환')


  # 레이아웃 조정
  plt.tight_layout()

  # 그래프 표시
  plt.show()
  ```
</details>

<br></br>

### ○ 20일 이동평균 주가
✔ SM 종목의 20일 이동평균 주가 움직임을 확인하였을 때, 최댓값과 최솟값이 감소하는 상황이며, 주가 상승과 하락도 극단적으로 줄어들고 있다.  
✔ 따라서, 최댓값과 최솟값의 감소는 전반적으로 추가가 하락하고 있는 추세를 나타낸다.

<img src='./images/20일이동평균.png' width='800'>
<details>
  <summary>Nan 제거 code</summary>

  ```
  # Nan 값 제거
  sm_df = pre_f_df[['SM']].dropna()
  sm_df
  ```
</details>
<details>
  <summary>이동평균 code</summary>

  ```
  # 윈도우 크기 지정
  window = 20

  # 이동평균 후 최솟값 계산
  sm_df['min'] = sm_df['SM'].rolling(window=window).min()
  # 이동평균 후 평균값 계산
  sm_df['mean'] = sm_df['SM'].rolling(window=window).mean()
  # 이동평균 후 최댓값 계산
  sm_df['max'] = sm_df['SM'].rolling(window=window).max()
  # 이동평균 후 중앙값 계산
  sm_df['median'] = sm_df['SM'].rolling(window=window).median()

  # 전체 대상으로 값을 구하는 것이 아니라 윈도우 값이 20이라고 가정하면 20 중 해당하는 값을 구하는 것
  # 최댓값을 구한다고 가정했을 때, 1~20 중 최댓값, 2~21 중 최댓값... 이런식으로 들감

  # Nan 값 제거
  sm_df.dropna()
  ```
</details>
<details>
  <summary>그래프 code</summary>

  ```
  import matplotlib.pyplot as plt

  # 최솟값, 평균값, 최댓값의 이동평균 시각화
  # 마지막 252일의 데이터 시각화(1년 치)
  # 최솟값, 최댓값 초록색 점선으로, 평균값은 빨간색으로 표기
  ax = sm_df[['min', 'mean', 'max', 'median']].iloc[-252:].plot(figsize=(15, 5), style=['g--', 'r--', 'g--', 'y--'], lw=0.8)
  # 마지막 252일의 원본 데이터도 함께 표기
  sm_df['SM'].iloc[-252:].plot(ax=ax)

  plt.title("SM 20-Day Moving Average Price Movement")
  plt.show()
  ```
</details>

<br></br>

### ○ 주가 기술분석
✔ 장기 선과 단기 선을 표시하여 골든/데드 크로스를 나타냈다.  
✔ 변화 폭이 가장 컸던 2023년 상반기는 장기가 상승하고 단기가 하락함에 따라 골든 크로스가 발생해 적극 매수가 권장됐고,  
  이후 2023년 하반기에 장기가 하락하고 단기가 상승함에 따라 데드 크로스가 발생해 적극 매도가 권장됐다.

<img src='./images/주가기술분석.png' width='800'>
<details>
  <summary>SMA code</summary>

  ```
  # SMA(Simple Moving Average): 일정 기간동안의 가격의 평균을 나타내는 보조지표
  # 1달 영업일을 21일로 가정, 1년 영업일을 252일로 가정

  sm_df['SMA1'] = sm_df['SM'].rolling(window=21).mean() #short-term (단기)
  sm_df['SMA2'] = sm_df['SM'].rolling(window=252).mean() #long-term (장기)
  sm_df[['SM', 'SMA1', 'SMA2']].tail()
  ```
</details>
<details>
  <summary>그래프 code</summary>

  ```
  # 주가 기술 분석
  # 골든 크로스, 데드 크로스

  # 데이터 NaN 값 제거
  sm_df.dropna(inplace=True)

  # 거래 신호 결정 기준 데이터 후 새로운 컬럼으로 추가
  # SMA1이 SMA2 보다 크면 1, 작으면 -1
  sm_df['positions'] = np.where(sm_df['SMA1'] > sm_df['SMA2'], 1, -1)  # 1: buy , -1: sell /

  # 주식 가격, 이동평균, 거래 신호 데이터 시각화
  # secondary_y: 보조 y축 지정 (오른쪽 표기)
  ax = sm_df[['SM', 'SMA1', 'SMA2', 'positions']].plot(figsize=(15, 5), secondary_y='positions')
  # 범례를 조정하여 그래프 우측 상단에 표시
  ax.get_legend().set_bbox_to_anchor((-0.05, 1))

  plt.title("SM Trading Window based on Technical Analysis")
  plt.show()
  ```
</details>

<br></br>
<br></br>
<br></br>
<br></br>
<br></br>

# Ⅲ. 머신러닝
## 1. 데이터 전처리
✔ 직관적인 컬럼명으로 변경

<details>
  <summary>code</summary>

  ```
  # 컬럼을 명확하게 확인할 수 있도록 컬럼명 변경
  pre_f_df = f_df.rename(columns={'Adj Close': 'SM'})
  pre_f_df
  ```
</details>

<br></br>

### ○ 결측치 제거
✔ 결측치 제거

<details>
  <summary>code</summary>

  ```
  # 결측치 제거
  pre_f_df = pre_f_df.dropna()
  pre_f_df
  ```
</details>

<br></br>

### ○ 특정 기간 데이터 추출
✔ 예측의 정확도를 높이기 위해 1년치 데이터 추출

<details>
  <summary>code</summary>

  ```
  # sm의 최근 1년간 데이터 추출
  sm_df = pre_f_df[['SM']].iloc[-252:]
  sm_df
  ```
</details>

<br></br>

### ○ 데이터 세트 분리
✔ 모델 평가를 위해 전체 데이터의 80%를 훈련 데이터로 나머지 20%를 평가 데이터로 분리했다.

<img src='./images/데이터분리.png' width='800'>

<details>
  <summary>데이터 분리code</summary>

  ```
  # 데이터 세트 분리 
  # 시계열 데이터에서 값을 랜덤하게 섞으면 시계열 데이터의 고유한 특성과 패턴이 손실되어 직접 데이터 세트를 분리해야 한다.

  # 앞쪽 80% 데이터를 훈련 데이터로 지정
  y_train = sm_df['SM'][:int(0.8 * len(sm_df))]

  # 뒷쪽 20% 데이터를 테스트 데이터로 지정
  y_test = sm_df['SM'][int(0.8 * len(sm_df)):]
  ```
</details>
<details>
  <summary>그래프 code</summary>

  ```
  import matplotlib.pyplot as plt

  plt.figure(figsize=(15, 5))

  # 분리된 데이터 시각화
  y_train.plot(color=colors[1])
  y_test.plot(color=colors[0])

  plt.show()
  ```
</details>

<br></br>
<br></br>

## 2. 머신러닝 훈련
### ○ ACF 및 PACF
✔ 좌측 ACF 그래프는 점차 감소하는 상관관계를 보이며, 느리게 감소함에 따라 비정상성을 띄고 있음을 확인할 수 있다.  
✔ 차분 후 우측 PACF 그래프에서 0에 거의 안착함에 따라 정상성을 가지고 있는 것을 확인할 수 있으며, 이는 AR(자기회귀) 모델을 적용하기에 적합하다.

<img src='./images/ACFPACF.png' width='800'>

<details>
  <summary>code</summary>

  ```
  import matplotlib.pyplot as plt
  from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

  # 그래프 사이즈 지정
  fig, ax = plt.subplots(1, 2, figsize=(15, 5))

  # 정상성 없는 데이터 acf, 차분된 데이터 pacf 계산 및 그래프 생성
  # 해당 그래프를 중점으로 확인하기
  plot_acf(sm_df, lags=20, ax=ax[0])
  plot_pacf(sm_df.diff().dropna(), lags=20, ax=ax[1])


  plt.show()
  ```
</details>

<br></br>

### ○ 차분 횟수
✔ d = 1  
✔ 검정을 통해 1차분이 가장 좋은 것으로 나타남

<details>
  <summary>code</summary>

  ```
  from pmdarima.arima import ndiffs

  # KPSS(Kwaiatkowski-Phillips-Schmidt-Shin)
  # ★ 차분을 진행하는 것이 필요할 지 결정하기 위해 사용하는 한 가지 검정 방법 ★
  # 영가설(귀무가설)을 "데이터에 정상성이 나타난다."로 설정한 뒤
  # 영가설이 거짓이라는 증거를 찾는 알고리즘이다.
  # 영가설(귀무가설): 통계적 가설 검정에서 처음으로 세우는 기본 가설, 검정하려는 주장이나 효과가 없다는 가정(두 변수간의 상관관계가 없음)

  # KPSS, ADF, PP 테스트를 통해 차분 횟수 계산
  kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=6)
  adf_diffs = ndiffs(y_train, alpha=0.05, test='adf', max_d=6)
  pp_diffs = ndiffs(y_train, alpha=0.05, test='pp', max_d=6)

  # 필요한 최적의 차분 횟수 계산
  n_diffs = max(kpss_diffs, adf_diffs, pp_diffs)

  # 최적의 차분 횟수 출력
  print(f'd = {n_diffs}')
  ```
</details>

<br></br>

### ○ 최적의 파라미터 확인
✔ Best model:  ARIMA(0,1,0)(0,0,0)[0] 
✔ 오토 아리마를 통해 최적의 파라미터 값을 도출함

<img src='./images/최적파라미터머신.png' width='500'>

<details>
  <summary>code</summary>

  ```
  import pmdarima as pm

  # 최적의 파라미터 값 계산
  # 예로, q 값을 가장 먼저 찾아내고 거기에 맞춰서 p 값을 찾아내는 식의 stepwise 단계별 알고리즘을 사용한다.
  model = pm.auto_arima(y=y_train, 
                        d=1, 
                        start_p=0, max_p=10, 
                        start_q=0, max_q=10, 
                        m=1, seasonal=False, 
                        stepwise=True, 
                        trace=True)

  # y = 훈련할 시계열 데이터
  # d = 차분횟수, 테스트 검정을 통해 값을 찾기(미설정 시 오래걸림)
  # start_p = AR 시작 차수, max_p = AR 최대 차수
  # start_q = MA 시작 차수, max_q = MA 최대 차수
  # m=계절성 (defalut=1, 1이면 계절성이 없다는 뜻으로 seasonal 미작성 가능)
  # seasonal=계절성 모델 여부 지정 (defalut=False)
  # stepwise=단계별 알고리즘 사용 여부
  # trace=모델 검색 과정 속 각 단계의 진행 상황 출력 여부 (cnn의 verbose와 유사)
  ```
</details>

<br></br>

### ○ 훈련

<img src='./images/머신러닝훈련.png' width='250'>

<details>
  <summary>code</summary>

  ```
  import pmdarima as pm

  # 최적의 파라미터 값 계산
  # 예로, q 값을 가장 먼저 찾아내고 거기에 맞춰서 p 값을 찾아내는 식의 stepwise 단계별 알고리즘을 사용한다.
  model = pm.auto_arima(y=y_train, 
                        d=1, 
                        start_p=0, max_p=10, 
                        start_q=0, max_q=10, 
                        m=1, seasonal=False, 
                        stepwise=True, 
                        trace=True)

  # y = 훈련할 시계열 데이터
  # d = 차분횟수, 테스트 검정을 통해 값을 찾기(미설정 시 오래걸림)
  # start_p = AR 시작 차수, max_p = AR 최대 차수
  # start_q = MA 시작 차수, max_q = MA 최대 차수
  # m=계절성 (defalut=1, 1이면 계절성이 없다는 뜻으로 seasonal 미작성 가능)
  # seasonal=계절성 모델 여부 지정 (defalut=False)
  # stepwise=단계별 알고리즘 사용 여부
  # trace=모델 검색 과정 속 각 단계의 진행 상황 출력 여부 (cnn의 verbose와 유사)
  ```
</details>

<br></br>
<br></br>

## 3. 모델 평가
### ○ 모델 정보 요약

<img src='./images/썸머리.png' width='550'>

<details>
  <summary>code</summary>

  ```
  # 모델의 요약 정보 출력
  print(model.summary())

  # Prob(Q), 융-박스 검정 통계량
  # 영가설: 잔차가 백색잡음 시계열을 따른다.
  # 0.05 이상: 서로 독립이고 동일한 분포를 따른다.

  # Prob(H), 이분산성 검정 통계량
  # 영가설: 잔차가 이분산성을 띠지 않는다.
  # 0.05 이상: 잔차의 분산이 일정하다.

  # Prob(JB), 자크-베라 검정 통계량
  # 영가설: 잔차가 정규성을 따른다.
  # 0.05 이상: 일정한 평균과 분산을 따른다.

  # Skew(쏠린 정도, 왜도)
  # 0에 가까워야 한다.

  # Kurtosis(뾰족한 정도, 첨도)
  # 3에 가까워야 한다.

  # 아마존 주식은 독립성을 보이나 수익률이 일정하지 않기 때문에 장기 보다는 중장기 방향성 또는 단기로 방향을 잡아야 한다. (Prob(JB)가 0.05 이상이라면 장기)
  # 즉, 지속적으로 동일한 수익률이 나타나기 어려워 보이며, 고위험군까지는 아니더라도 중립이거나 위험도가 조금 있을 것이다.
  # 이러한 내용을 토대로 투자 전략을 세울 수 있다.
  ```
</details>

<br></br>

### ○ 모델 진단 그래픽

<img src='./images/진단그래픽.png' width='800'>

<details>
  <summary>code</summary>

  ```
  import matplotlib.pyplot as plt

  # 모델 진단 그래픽
  model.plot_diagnostics(figsize=(16, 8))
  plt.show()
  ```
</details>

<br></br>

### ○ 분석
✔ Prob(Q), 융-박스 검정 통계량 수치가 0.58로 나타났고,  
Correlogram(코렐로그램)에서도 0 주변에 안착(정상 시계열) 했기 때문에 잔차가 독립적이라고 보여진다.  
따라서 시계열 데이터에서의 자기상관 구조가 없어 보인다.
  
✔ Prob(H), 이분산성 검정 통계량 수치가 0.00으로 나타났고,  
Standardized residual(스탠다다이즈드 레지듀얼스)에서 잔차의 분산이 일정하지 않음에 따라 이분산성이 있다고 보여진다.

✔ Prob(JB), 자크-베라 검정 통계량 수치가 0.00으로 나타났고,  
Histogram(히스토그램)에서 팻 테일 리스크가 보이며  
Normal Q-Q(노멀 큐-큐)에서 잔차가 45도 선상에 분포되어 있지 않기 때문에 정규분포를 따르지 않는다고 보여진다.

✔ Skew, 왜도 수치는 0.62으로 나타났고 Kurtosis, 첨도 수치는 6.30으로 나타났다.  
Histogram(히스토그램)에서 종목인 KDE가 정규분포인 N보다 조금 더 좌측으로 쏠리는 오른쪽으로 꼬리가 긴 형태이며,  
첨도는 뾰족한 것으로 보여진다.

<br></br>
<br></br>

## 4. 예측
### ○ 예측

|  일자 | 예측값     |
|:---:|:-------:|
| 201 | 87800.0 |
| 202 | 87800.0 |
| 203 | 87800.0 |
| ... | ...     |
| 249 | 87800.0 |
| 250 | 87800.0 |
| 251 | 87800.0 |

<details>
  <summary>code</summary>

  ```
  # 예측
  # n_periods: 예측 기간 지정
  prediction = model.predict(n_periods=len(y_test))
  prediction

  # len(y_test)만큼의 기간 동안의 예측을 수행한다.
  # 예측을 기반으로 주어진 입력 데이터의 패턴 및 동향을 고려하여 값을 예상하는데,
  # 새로운 데이터가 들어오면 해당 데이터를 사용하여 모델을 업데이트해야만 추가적인 예측이 수행된다.
  ```
</details>

<br></br>

### ○ 분석
✔ y_test 길이만큼의 기간 동안의 예측을 수행 시  
  주어진 입력 데이터의 패턴 및 동향을 고려하여 값을 예상하는데,  
  모델이 현재 사용 중인 데이터에 대해서만 학습하기 때문에 동일한 값만 나타나고 있다.

✔ 따라서, 모델 업데이트를 해야만 추가적인 예측이 제대로 수행된다.

<br></br>

### ○ update를 통한 예측

<img src='./images/업뎃예측.png' width='800'>

<details>
  <summary>예측 함수 선언 code</summary>

  ```
  # 예측 함수 선언
  def predict_one_step():
      prediction = model.predict(n_periods=1)
      return prediction.tolist()[0]
  ```
</details>
<details>
  <summary>예축 code</summary>

  ```
  # 예측값 담을 초기 list 선언 (시각화를 위함)
  p_list = []

  for data in y_test:
      # 예측 함수로 예측값 가져오기
      p = predict_one_step()
      # 예측값 저장
      p_list.append(p)

      # 모델 업데이트
      model.update(data)
  ```
</details>
<details>
  <summary>데이터프레임 변환 code</summary>

  ```
  # test 값과 예측값을 데이터 프레임으로 생성
  y_predict_df = pd.DataFrame({"test": y_test, "pred": p_list})
  y_predict_df
  ```
</details>
<details>
  <summary>그래프 code</summary>

  ```
  import matplotlib.pyplot as plt

  # 그래프 크기 지정
  fig, ax = plt.subplots(1, 1, figsize=(12, 6))

  # 특정 train 데이터
  plt.plot(y_train.iloc[-50:], label='Train', color=colors[1])
  # 특정 test 데이터
  plt.plot(y_test.iloc[-50:], label='Test', color=colors[4])
  # 예측 데이터
  plt.plot(y_predict_df.pred, label='Prediction', color=colors[0])

  plt.legend()
  plt.show()
  ```
</details>

<br></br>

### ○ update를 통한 예측
✔ MAPE (%): 2.6830

<details>
  <summary>code</summary>

  ```
  import numpy as np

  # 평균 오차율 계산 함수 선언
  def MAPE(y_test, y_pred):
      # 각 예측값에 대한 상대적인 오차율을 계산
      # abs: 절댓값 계산
      return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

  # 함수를 통해 오차율 출력
  print(f'MAPE (%): {MAPE(y_test, p_list):.4f}')
  ```
</details>

<br></br>

### ○ 분석
✔ 실제 값을 알고 있어야 예측이 되기 때문에 한 스텝 씩 업데이트를 하여 예측한 결과  
  2.6830% 센트의 오차가 있으나 시각화 자료를 확인하였을 때, 거의 유사한 것을 알 수 있었다.

✔ 시계열 데이터는 실제 데이터와 모델의 예측 값을 비교하여 모델의 평가를 위해 사용되는 것으로 미래이의 값을 예측하기는 어렵다.  
  따라서 딥러닝에서 미래를 예측해보기로 한다.

<br></br>
<br></br>
<br></br>
<br></br>
<br></br>

# Ⅳ. 딥러닝
## 1. 1cycle
### ○ 최적의 파라미터
| changepoint_prior_scale | seasonality_prior_scale | seasonality_mode | mape     |
|:-----------------------:|:-----------------------:|:----------------:|:--------:|
| 0.10                    | 0.10                    | additive         | 0.084216 |

<details>
  <summary>Prophet code</summary>

  ```
  from prophet import Prophet
  from prophet.diagnostics import cross_validation, performance_metrics
  import itertools

  # 파라미터 값 지정
  # changepoint_prior_scale: trend의 변화하는 크기를 반영하는 정도이다, 0.05가 default (10.0 이상은 비추천)
  # seasonality_prior_scale: 계절성을 반영하는 단위이다.
  # seasonality_mode: 계절성으로 나타나는 효과를 더해 나갈지, 곱해 나갈지 정한다.
  # additive: 더하기, multiplicative: 곱하기
  search_space = {
      'changepoint_prior_scale': [0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
      'seasonality_prior_scale': [0.05, 0.1, 1.0, 10.0],
      'seasonality_mode': ['additive', 'multiplicative']
  }

  # itertools.product(): 각 요소들의 모든 경우의 수 조합으로 생성
  # *unpacking으로 값을 풀어 가져오고 zip을 사여 key와 묶어 dict 형식으로 저장  
  param_combinded = [dict(zip(search_space.keys(), v)) for v in itertools.product(*search_space.values())]

  # 전체 데이터 80% 개수 저장
  train_len = int(len(sm_df) * 0.8)
  # 전체 데이터 20% 개수 저장
  test_len = int(len(sm_df) * 0.2)

  # 훈련 사이즈 지정
  train_size = f'{train_len} days'
  # 테스트 사이즈 지정
  test_size = f'{test_len} days'
  # 훈련 데이터 세트 분리 (상위 80%)
  train_df = sm_df.iloc[: train_len]
  # 테스트 데이터 세트 분리 (하위 20%)
  test_df = sm_df.iloc[train_len: ]

  # 평균 절대 백분율 오차(MAPE) 저장하기 위해 초기 list 생성
  mapes = []

  for param in param_combinded:
      # 파라미터 값이 dict 형식이기 때문에 unpacking하여 Prophet 모델에 전달
      model = Prophet(**param)
      # 훈련
      model.fit(train_df)

      # cross_validation
      # initial=초기 학습 기간, period=교차 검증을 수행할 각 반복의 기간, horizon=예측할 기간, parallel=병렬 처리를 사용하여 교차 검증 수행 여부
      # parallel 옵션은 아래와 같다.
      # 'threads' 옵션은 메모리 사용량은 낮지만 CPU 바운드 작업에는 효과적이지 않을 수 있다.
      # 'dask' 옵션은 대규모의 데이터를 처리하는 데 효과적이다.
      # 'processes' 옵션은 각각의 작업을 별도의 프로세스로 실행하기 때문에 CPU 바운드 작업에 효과적이지만,
      # 메모리 사용량이 높을 수 있다.
      cv_df = cross_validation(model, initial=train_size, period='20 days', horizon=test_size, parallel='processes')
      # 교차 검증 결과를 평가
      # performance_metrics(교차 검증을 수행한 후 얻은 데이터프레임, rolling_window=이동 평균 계산)
      df_p = performance_metrics(cv_df, rolling_window=1)
      # 데이터 프레임에서 mape 값을 추출하여 list에 저장
      # mape이 제일 낮은게 우리 가 사용해야할 파라미터 값이 된다.
      mapes.append(df_p['mape'].values[0])

  # 매개변수 조합을 포함하는 데이터프레임 생성
  tuning_result = pd.DataFrame(param_combinded)
  # 평균 절대 백분율 오차(MAPE) 값을 컬럼으로 추가
  tuning_result['mape'] = mapes
  ```
</details>
<details>
  <summary>파라미터 값 순위 code</summary>

  ```
  tuning_result.sort_values(by='mape')
  ```
</details>

<br></br>

### ○ 훈련
| index | ds         | yhat          | yhat_lower   | yhat_upper     |
|:-----:|:----------:|:-------------:|:------------:|:--------------:|
| 0     | 2023-06-05 | 101742.962992 | 96147.472916 | 107747.209138  |
| 1     | 2023-06-07 | 103372.040150 | 98176.721962 | 109333.865096  |
| 2     | 2023-06-08 | 103523.087639 | 98025.281060 | 109339.006053  |
| ...   | ...        | ...           | ...          | ...            |
| 614   | 2025-06-15 | 119189.459112 | -4517.187385 | 248721.952052  |
| 615   | 2025-06-16 | 118889.465074 | -6011.385966 | 252070.115030  |
| 616   | 2025-06-17 | 119565.053464 | -4991.131407 | 252204.959065  |

<details>
  <summary>code</summary>

  ```
  # loss 값이 제일 낮은 파라미터 값 가져와서 담기 
  # 최적의 모델!
  model = Prophet(changepoint_prior_scale=0.10, 
                  seasonality_prior_scale=0.10, 
                  seasonality_mode='additive')

  # 훈련
  model.fit(sm_df)

  # Prophet 모델을 사용하여 미래의 예측값을 생성한다.
  # make_future_dataframe: 미래의 일정한 기간에 해당하는 날짜를 포함하는 DataFrame 생성
  future = model.make_future_dataframe(periods=365)

  # 예측
  # 미래의 날짜 정보가 포함된 future DataFrame 사용
  forecast = model.predict(future)
  # 예측 결과 중 특정 컬럼 추출하여 출력
  # ds: 날짜, yhat: 해당 날짜의 예측값, yhat_lower: 예측값 하한, yhat_upper: 예측값 상한
  forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
  ```
</details>

<br></br>

### ○ SM 엔터테인먼트 주가 미래 예측

<img src='./images/1cycle.png' width='800'>
<img src='./images/산점도.png' width='800'>

<details>
  <summary>시계열 데이터프레임 변환 code</summary>

  ```
  # 데이터 프레임 복제
  pre_sm_df = sm_df.copy()
  # 특정 컬럼을 인덱스로 지정
  pre_sm_df.set_index('ds', inplace=True)

  # forecast(예측) 데이터 프레임 복사
  forecast_df = forecast.copy()
  # 특정 컬럼을 인덱스로 지정
  forecast_df = forecast_df.set_index('ds')

  # 인덱스를 datetime 형식으로 변환
  pre_sm_df.index = pd.to_datetime(pre_sm_df.index)
  forecast_df.index = pd.to_datetime(forecast_df.index)
  ```
</details>
<details>
  <summary>그래프 code</summary>

  ```
  # 그래프 사이즈 지정
  fig, ax = plt.subplots(1, 1, figsize=(12, 6))

  # 훈련 데이터
  plt.plot(pre_sm_df[['y']], label='Train', color=colors[1])
  # 예측 데이터
  plt.plot(forecast_df[['yhat']], label='Prediction', color=colors[0])

  plt.legend()
  plt.show()
  ```
</details>
<details>
  <summary>산점도 그래프 code</summary>

  ```
  # 예측 결과 데이터 프레임의 정보로 시각화
  model.plot(forecast, figsize=(15, 8), xlabel='year-month', ylabel='price')

  plt.show()
  ```
</details>

<br></br>

### ○ 분석
✔ 예측 결과 신뢰 구간이 뒤로 갈수록 점점 넓어지는 것을 확인할 수 있었으며,  
  훈련 값이 신뢰 구간에서 조금씩 벗어나는 것으로 보여졌다.

✔ 따라서, 1년치 데이터로는 정확한 예측이 불가능하다고 판단되어 3년치 데이터로 예측을 시도해보기로 한다.

<br></br>

### ○ 기간 별 예측
✔ 월간

<img src='./images/1월간.png' width='800'>

<br>

✔ 주간

<img src='./images/1주간.png' width='800'>

<br></br>

### ○ 분석
<img src='./images/sm12.png' width=500px>

<br>

✔ 연간 그래프를 보았을 때 2023년 11월 쯤 주가가 크게 하락하는데, 그룹 라이즈의 멤버 홍승한 군의 사생활 논란과 맞물린다.

✔ 이후 점차 소폭 상승하는 추세를 보인다.

✔ 주간 그래프 확인 시 월요일에 많이 하락하며 수요일에 많이 상승하는 것을 알 수 있었다.

<br></br>
<br></br>

## 2. 2cycle
### ○ 데이터 추출
✔ 3년간 데이터 추출

<details>
  <summary>code</summary>

  ```
  # sm의 최근 3년간 데이터 추출
  sm2_df = pre_f_df.iloc[-756:].reset_index(drop=True)
  sm2_df
  ```
</details>

<br></br>

### ○ 최적의 파라미터
| changepoint_prior_scale | seasonality_prior_scale | seasonality_mode | mape     |
|:-----------------------:|:-----------------------:|:----------------:|:--------:|
| 5.00                    | 0.05                    | multiplicative         | 0.190932 |

<details>
  <summary>Prophet code</summary>

  ```
  from prophet import Prophet
  from prophet.diagnostics import cross_validation, performance_metrics
  import itertools

  # 파라미터 값 지정
  # changepoint_prior_scale: trend의 변화하는 크기를 반영하는 정도이다, 0.05가 default (10.0 이상은 비추천)
  # seasonality_prior_scale: 계절성을 반영하는 단위이다.
  # seasonality_mode: 계절성으로 나타나는 효과를 더해 나갈지, 곱해 나갈지 정한다.
  # additive: 더하기, multiplicative: 곱하기
  search_space = {
      'changepoint_prior_scale': [0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
      'seasonality_prior_scale': [0.05, 0.1, 1.0, 10.0],
      'seasonality_mode': ['additive', 'multiplicative']
  }

  # itertools.product(): 각 요소들의 모든 경우의 수 조합으로 생성
  # *unpacking으로 값을 풀어 가져오고 zip을 사여 key와 묶어 dict 형식으로 저장  
  param_combinded = [dict(zip(search_space.keys(), v)) for v in itertools.product(*search_space.values())]

  # 전체 데이터 80% 개수 저장
  train_len = int(len(sm2_df) * 0.8)
  # 전체 데이터 20% 개수 저장
  test_len = int(len(sm2_df) * 0.2)

  # 훈련 사이즈 지정
  train_size = f'{train_len} days'
  # 테스트 사이즈 지정
  test_size = f'{test_len} days'
  # 훈련 데이터 세트 분리 (상위 80%)
  train_df = sm2_df.iloc[: train_len]
  # 테스트 데이터 세트 분리 (하위 20%)
  test_df = sm2_df.iloc[train_len: ]

  # 평균 절대 백분율 오차(MAPE) 저장하기 위해 초기 list 생성
  mapes = []

  for param in param_combinded:
      # 파라미터 값이 dict 형식이기 때문에 unpacking하여 Prophet 모델에 전달
      model = Prophet(**param)
      # 훈련
      model.fit(train_df)

      # cross_validation
      # initial=초기 학습 기간, period=교차 검증을 수행할 각 반복의 기간, horizon=예측할 기간, parallel=병렬 처리를 사용하여 교차 검증 수행 여부
      # parallel 옵션은 아래와 같다.
      # 'threads' 옵션은 메모리 사용량은 낮지만 CPU 바운드 작업에는 효과적이지 않을 수 있다.
      # 'dask' 옵션은 대규모의 데이터를 처리하는 데 효과적이다.
      # 'processes' 옵션은 각각의 작업을 별도의 프로세스로 실행하기 때문에 CPU 바운드 작업에 효과적이지만,
      # 메모리 사용량이 높을 수 있다.
      cv_df = cross_validation(model, initial=train_size, period='20 days', horizon=test_size, parallel='processes')
      # 교차 검증 결과를 평가
      # performance_metrics(교차 검증을 수행한 후 얻은 데이터프레임, rolling_window=이동 평균 계산)
      df_p = performance_metrics(cv_df, rolling_window=1)
      # 데이터 프레임에서 mape 값을 추출하여 list에 저장
      # mape이 제일 낮은게 우리 가 사용해야할 파라미터 값이 된다.
      mapes.append(df_p['mape'].values[0])

  # 매개변수 조합을 포함하는 데이터프레임 생성
  tuning_result = pd.DataFrame(param_combinded)
  # 평균 절대 백분율 오차(MAPE) 값을 컬럼으로 추가
  tuning_result['mape'] = mapes
  ```
</details>
<details>
  <summary>파라미터 값 순위 code</summary>

  ```
  tuning_result.sort_values(by='mape')
  ```
</details>

<br></br>

### ○ 분석
```
'changepoint_prior_scale': [0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
'seasonality_prior_scale': [0.05, 0.1, 1.0, 10.0],
'seasonality_mode': ['additive', 'multiplicative']
```

<br>

✔ 위 파라미터 값과 3년치 데이터를 통해 최적의 파라미터 값을 탐색한 결과 로스 값 약 0.1911으로 나오는  
changepoint_prior_scale=5.00,  
seasonality_prior_scale=0.05,  
seasonality_mode='multiplicative'이 가장 최적의 파라미터로 나타났으나,  
multiplicative의 주기성을 더 주려고 하는 특성으로 아래와 같은 그래프가 그려지게 됐다.

<br>

<img src='./images/deep01.png' width=800px>

<br>

✔ 따라서, 기존 파라미터 값에 모드만 'additive'로 변경하여 사용하고자 한다.

<br></br>

### ○ 훈련
| index | ds         | yhat          | yhat_lower    | yhat_upper    |
|:-----:|:----------:|:-------------:|:-------------:|:-------------:|
| 0     | 2021-05-17 | 33514.257466  | 2.776573e+04  | 3.906994e+04  |
| 1     | 2021-05-18 | 34990.109701  | 2.927235e+04  | 4.032550e+04  |
| 2     | 2021-05-20 | 36313.174393  | 3.106384e+04  | 4.184090e+04  |
| ...   | ...        | ...           | ...           | ...           |
| 1509  | 2026-07-11 | -47107.224017 | -1.691377e+06 | 1.626280e+06  |
| 1510  | 2026-07-12 | -47155.570207 | -1.697815e+06 | 1.623643e+06  |
| 1511  | 2026-07-13 | -47848.236135 | -1.702218e+06 | 1.625307e+06  |

<details>
  <summary>code</summary>

  ```
  # loss 값이 제일 낮은 파라미터 값 가져와서 담기 
  # 최적의 모델!
  model = Prophet(changepoint_prior_scale=5.00, 
                  seasonality_prior_scale=0.05, 
                  seasonality_mode='additive')

  # 훈련
  model.fit(sm2_df)

  # Prophet 모델을 사용하여 미래의 예측값을 생성한다.
  # make_future_dataframe: 미래의 일정한 기간에 해당하는 날짜를 포함하는 DataFrame 생성
  future = model.make_future_dataframe(periods=756)

  # 예측
  # 미래의 날짜 정보가 포함된 future DataFrame 사용
  forecast = model.predict(future)
  # 예측 결과 중 특정 컬럼 추출하여 출력
  # ds: 날짜, yhat: 해당 날짜의 예측값, yhat_lower: 예측값 하한, yhat_upper: 예측값 상한
  forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
  ```
</details>

<br></br>

### ○ SM 엔터테인먼트 주가 미래 예측

<img src='./images/2cycle.png' width='800'>
<img src='./images/2산점도.png' width='800'>

<details>
  <summary>시계열 데이터프레임 변환 code</summary>

  ```
  # 데이터 프레임 복제
  pre_sm2_df = sm2_df.copy()
  # 특정 컬럼을 인덱스로 지정
  pre_sm2_df.set_index('ds', inplace=True)

  forecast_df = forecast.copy()
  forecast_df = forecast_df.set_index('ds')

  pre_sm2_df.index = pd.to_datetime(pre_sm2_df.index)
  forecast_df.index = pd.to_datetime(forecast_df.index)
  ```
</details>
<details>
  <summary>그래프 code</summary>

  ```
  import matplotlib.pyplot as plt

  # 그래프 사이즈 지정
  fig, ax = plt.subplots(1, 1, figsize=(12, 6))

  # 훈련 데이터
  plt.plot(pre_sm2_df[['y']], label='Train', color=colors[1])
  # 예측 데이터
  plt.plot(forecast_df[['yhat']], label='Prediction', color=colors[0])

  plt.legend()
  plt.show()
  ```
</details>
<details>
  <summary>산점도 그래프 code</summary>

  ```
  # 예측 결과 데이터 프레임의 정보로 시각화
  model.plot(forecast, figsize=(15, 8), xlabel='year-month', ylabel='price')

  plt.show()
  ```
</details>

<br></br>

### ○ 분석
✔ 예측 결과 신뢰 구간이 뒤로 갈수록 점점 넓어지는 것을 확인할 수 있었으며, 훈련 값이 신뢰 구간과 거의 유사한 것으로 보여졌다.

<br></br>

### ○ 기간 별 예측
✔ 연간

<img src='./images/2연간.png' width='800'>

<br>

✔ 월간

<img src='./images/2월간.png' width='800'>

<br>

✔ 주간

<img src='./images/2주간.png' width='800'>

<br></br>

### ○ 분석
<img src='./images/sm12.png' width=500px>

<br>

✔ 연간 그래프를 보았을 때 거의 일정하게 나오지만 소폭 하락하는 추세가 보여진다.

✔ 주간 그래프 확인 시 월요일에 많이 하락하며 수요일에 많이 상승하는 것을 알 수 있었다.

✔ 월간 그래프 확인 시 2월과 11월에 많이 하락하고 5월부터 8월까지 상승하는 추세가 보여진다.

<br></br>
<br></br>
<br></br>
<br></br>
<br></br>

# Ⅴ. 결론
✔ SM은 변동률이 높으므로 안정성을 추구하는 투자자들에게는 적합하지 않을 수 있다.

<br>

<img src='./images/t01.png' width=800px>

<br>

✔ 주간 그래프에서 월요일에 하락하고 화요일에 오르는 것으로 나타났는데,   
실제 오늘(240618, 화) SM의 주가가 상승했다.

✔ 앞으로의 주가는 하락한다고 예측되나,  
4분기 신규 걸그룹 론칭 성공 여부에 따라 주가 변동이 있을 것이라 생각된다(쇼크).

✔ 현재 데드 크로스가 나타나므로 매수를 권장하지 않는다.