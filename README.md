# 준지도 학습 기반 노인 음성 인식 모델 학습

> Temporal Ensembling을 활용한 저비용 노인 헬스 케어 보조용 ASR 모델 학습 프로젝트  
> 50개의 음성-전사 데이터와 150개의 비전사 음성 데이터를 활용해 Whisper 기반 노인 음성 인식 성능을 개선했습니다.

---

## 1. Project Overview

본 프로젝트는 노인 헬스 케어 환경에서 사용할 수 있는 한국어 노인 음성 인식 모델을 학습한 프로젝트입니다.

노인 음성은 조음 기관의 노화, 발음 변화, 방언, 휴지, 발화 속도 차이 등으로 인해 일반 성인 음성보다 인식 난도가 높습니다. 또한 노인 음성 데이터는 구축 비용이 크고, 전사 데이터 확보가 어렵다는 한계가 있습니다.

이 문제를 해결하기 위해 본 프로젝트에서는 OpenAI Whisper 모델을 기반으로, 적은 양의 라벨링 데이터와 추가 비라벨 음성 데이터를 함께 활용하는 준지도 학습 방식을 적용했습니다. 특히 Temporal Ensembling 방법론을 사용하여 labeled data와 unlabeled data를 함께 학습하고, 노인 음성 인식에서 CER과 WER을 낮추는 것을 목표로 했습니다.

---

## 2. Motivation

노인 간병 및 헬스 케어 서비스에서는 문진, 복약 확인, 증상 기록, 간병 일지 작성 등 음성을 텍스트로 변환해야 하는 상황이 많습니다.

그러나 실제 현장에서는 다음과 같은 문제가 존재합니다.

- 노인 음성은 청장년 음성보다 인식 정확도가 낮음
- 발음 변화, 방언, 어절 경계 오류, 음운 변동으로 전사 오류가 발생함
- 노인 음성 데이터는 상대적으로 부족함
- 음성-전사 쌍 데이터를 구축하는 데 많은 비용과 시간이 필요함
- 대규모 ASR 모델을 학습하기 위한 GPU 비용 부담이 큼

따라서 본 프로젝트는 소량의 전사 데이터만으로도 노인 음성 인식 모델을 개선할 수 있는 저비용 학습 방식을 실험했습니다.

---

## 3. Dataset

본 프로젝트에서는 AI Hub의 노인 및 의료 도메인 음성 데이터를 활용했습니다.

| Dataset | Usage |
|---|---|
| 자유대화 음성(노인남여) | labeled data로 사용 |
| 비대면 진료를 위한 의료진 및 환자 음성 | 환자 음성을 unlabeled/validation data로 사용 |

데이터 구성은 다음과 같습니다.

| Type | Size | Description |
|---|---:|---|
| Labeled Data | 50 | 음성과 전사 텍스트가 쌍으로 존재하는 노인 음성 데이터 |
| Unlabeled Data | 150 | 전사 텍스트 없이 음성만 사용하는 노인 환자 음성 데이터 |
| Validation Data | 별도 구성 | 비대면 진료 데이터 중 검증용 환자 음성 |

전처리 과정에서는 다음 작업을 수행했습니다.

- 결측치 제거
- 비문자열 데이터 제거
- 빈 문자열 제거
- 중복 공백 제거
- 괄호 안 삽입 기호 제거
- 줄바꿈 및 불필요한 기호 제거
- 음성 파일을 16kHz로 로드
- Whisper 입력 형식에 맞게 padding 및 log-mel spectrogram 변환

---

## 4. Model

본 프로젝트에서는 OpenAI Whisper 모델을 사용했습니다.

실험한 모델 크기는 다음과 같습니다.

| Model Size | GPU |
|---|---|
| Whisper tiny | V100 |
| Whisper base | V100 |
| Whisper small | V100 |
| Whisper medium | A100 |

GPU 사용량을 고려하여 Whisper large 모델은 실험에서 제외했습니다.

---

## 5. Method

### 5.1 Baseline Fine-tuning

baseline 모델은 labeled data만 사용하여 supervised learning 방식으로 fine-tuning했습니다.

학습 과정은 다음과 같습니다.

1. 음성 파일 로드
2. 16kHz sampling rate로 변환
3. Whisper 입력에 맞게 log-mel spectrogram 생성
4. 전사 텍스트 tokenization
5. encoder-decoder 구조로 전사 예측
6. CrossEntropyLoss 기반 학습
7. CER, WER로 성능 평가

baseline은 Temporal Ensembling 방식의 효과를 비교하기 위한 기준 모델로 사용했습니다.

---

### 5.2 Temporal Ensembling

Temporal Ensembling은 labeled data와 unlabeled data를 함께 활용하는 준지도 학습 방법입니다.

본 프로젝트에서는 labeled data에 대해서는 정답 전사 텍스트를 기준으로 supervised loss를 계산하고, unlabeled data에 대해서는 이전 epoch의 예측값과 현재 epoch의 예측값 사이의 차이를 줄이도록 unsupervised loss를 계산했습니다.

최종 loss는 다음과 같이 구성했습니다.

```text
Total Loss = Supervised Loss + w * Unsupervised Loss
```

- Supervised Loss: labeled data에 대한 CrossEntropyLoss
- Unsupervised Loss: 이전 예측값과 현재 예측값 사이의 MSE Loss
- w: epoch와 labeled/unlabeled data 비율을 고려한 가중치

비지도 학습 데이터에 대한 영향력을 조절하기 위해 `max_val` 값을 0.1, 0.5, 0.8로 변경하며 실험했습니다.

---

## 6. Audio Augmentation

노인 음성의 특징을 반영하기 위해 음성 데이터에 여러 augmentation을 적용했습니다.

사용한 augmentation은 다음과 같습니다.

| Method | Description |
|---|---|
| Pitch Shift | 음성 pitch를 낮춰 노인 음성의 특성을 반영 |
| White Noise | 난수 기반 noise를 추가하여 음성 변동성 반영 |
| Time Stretch | 발화 속도 변화 반영 |
| Dropout | 일부 음성 신호를 제거하여 잡음 환경 반영 |
| Amplitude Modulation | 진폭을 주기적으로 변형하여 음성 떨림 특성 반영 |
| Mixed Augmentation | pitch shift, white noise, modulation 등을 조합 |

최종 실험에서는 pitch를 2 steps 낮추고, white noise와 amplitude modulation을 함께 적용하는 방식도 사용했습니다.

---

## 7. Training Settings

주요 학습 설정은 다음과 같습니다.

| Parameter | Value |
|---|---|
| Epochs | 10 |
| Labeled Data Size | 50 |
| Unlabeled Data Size | 150 |
| Batch Size | 2 |
| Learning Rate | 2e-5 |
| Max Length | 100 |
| max_val | 0.1, 0.5, 0.8 |
| Optimizer | AdamW |
| Loss | CrossEntropyLoss + MSE Loss |
| Metrics | CER, WER |

---

## 8. Evaluation Metrics

성능 평가는 CER과 WER을 기준으로 진행했습니다.

| Metric | Description |
|---|---|
| CER | Character Error Rate, 문자 단위 오류율 |
| WER | Word Error Rate, 단어 단위 오류율 |

CER과 WER은 낮을수록 좋은 성능을 의미합니다.

---

## 9. Additional Test Result

학습된 Whisper small 모델을 사용해 비대면 진료 환자 음성 validation data 3,879개에 대해 추가 평가를 수행했습니다.

| Model | Test Data | CER | WER |
|---|---|---:|---:|
| Whisper small + TE | 환자 음성 3,879개 | 18.01 | 48.52 |

기존 연구에서 언급된 네이버 클로바 API의 노인 음성 CER 63.21%와 비교했을 때, 본 실험의 CER 18.01%는 노인 헬스 케어 도메인 음성 인식 가능성을 보여주는 결과였습니다.

---

## 10. Error Analysis

전사 오류는 다음과 같은 유형으로 나타났습니다.

### 10.1 Phonetic Errors

노인 음성과 방언 발화에서는 특정 모음이나 자음이 표준 발음과 다르게 실현되며, 모델이 이를 발음 그대로 전사하는 경우가 있었습니다.

예를 들어 경상권 화자의 경우, 특정 모음이 다른 모음으로 실현되는 현상이 나타났고, 이로 인해 실제 의미와 다른 전사 결과가 생성되었습니다.

### 10.2 Phonological Errors

비음화와 같은 음운 변동이 표기 형태로 복원되지 않고, 발음에 가까운 형태로 전사되는 오류가 발생했습니다.

### 10.3 Morphological Errors

표기와 발음이 일치하지 않는 단어, 구어체 표현, 띄어쓰기 오류, 어절 경계 인식 오류가 나타났습니다.

### 10.4 Syntactic Errors

연결어미와 종결어미를 혼동하여 문장 구조가 달라지는 오류가 발생했습니다. 이는 음성적으로 유사한 어미를 구분하는 데 모델이 어려움을 보였기 때문으로 해석됩니다.

### 10.5 Numeric Expression Errors

숫자를 한글로 전사해야 하는 경우와 아라비아 숫자로 전사하는 경우가 혼재되어, 의미는 동일하지만 정량 평가에서는 오류로 계산되는 사례가 있었습니다.

---

## 11. Project Structure

```bash
.
├── train_whisper_sl.py
├── train_whisper_ssl.py
├── test_whisper_ssl.py
├── 모듈화_baseline.ipynb
└── 모듈화_temporal_ensembling_label_100.ipynb
```

| File | Description |
|---|---|
| train_whisper_sl.py | labeled data만 사용하는 baseline supervised learning 학습 코드 |
| train_whisper_ssl.py | Temporal Ensembling 기반 semi-supervised learning 학습 코드 |
| test_whisper_ssl.py | 학습된 Whisper 모델의 CER/WER 평가 코드 |
| 모듈화_baseline.ipynb | baseline 실험 실행 노트북 |
| 모듈화_temporal_ensembling_label_100.ipynb | Temporal Ensembling 실험 실행 노트북 |

---

## 12. Tech Stack

| Category | Stack |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| ASR Model | OpenAI Whisper |
| Audio Processing | librosa |
| NLP/ASR Utilities | Hugging Face Transformers, jiwer, evaluate |
| Optimizer | AdamW |
| Experiment Environment | Google Colab, V100, A100 |

---

## 13. How to Run

### 13.1 Baseline Training

```python
import train_whisper_sl

results = train_whisper_sl.train_sl(
    seed=seed,
    meta_data=meta_data,
    train_df=train_df,
    valid_df=valid_df,
    all_results=all_results
)
```

### 13.2 Temporal Ensembling Training

```python
import train_whisper_ssl

results = train_whisper_ssl.train_ssl(
    seed=seed,
    max_val=max_val,
    mult=mult,
    meta_data=meta_data,
    train_df=train_df,
    valid_df=valid_df,
    all_results=all_results
)
```

### 13.3 Evaluation

```python
import test_whisper_ssl

test_results = test_whisper_ssl.test_ssl(
    seed=seed,
    meta_data=meta_data,
    model_path=model_path,
    test_df=test_df,
    all_results=all_results
)
```

---

## 14. My Contribution

본 프로젝트에서 담당한 역할은 다음과 같습니다.

- 노인 음성 인식 문제 정의 및 실험 설계
- AI Hub 기반 노인 음성/의료 음성 데이터 구성
- 음성-전사 데이터 전처리 파이프라인 구현
- Whisper 기반 baseline fine-tuning 코드 구현
- Temporal Ensembling 기반 준지도 학습 코드 구현
- labeled/unlabeled data를 구분하는 custom dataset 설계
- pitch shift, white noise, amplitude modulation 등 audio augmentation 구현
- CER, WER 기반 성능 평가 코드 구현
- 모델 크기별 tiny/base/small/medium 성능 비교
- max_val 변화에 따른 비지도 loss 가중치 실험
- 전사 결과 오류 유형 분석

---

## 15. Limitations

본 프로젝트는 적은 수의 labeled data만으로 노인 음성 인식 모델을 개선하려는 시도였기 때문에 다음과 같은 한계가 있습니다.

- labeled data 수가 50개로 매우 적어 모델 학습 안정성이 낮음
- 모델 크기가 커질수록 일부 구간에서 CER, WER, loss가 급격히 증가함
- max_val이 낮은 경우 unsupervised loss 반영이 불안정하게 작동함
- 방언 발화자의 경우 표준어로 복원하지 못하고 발음 그대로 전사하는 오류가 발생함
- 정답 데이터 자체에 구어체 표기, 숫자 표기, 띄어쓰기 차이가 포함되어 정량 평가가 실제 의미 전달력을 완전히 반영하지 못함
