## Open Domain Question Answering (ODQA)
사전에 구축되어 있는 Knowledge resource에서 사용자의 질문과 관련된 문서를 추출하고 해당 문서에서 질문의 정답을 찾는 task이다.
retriever와 reader, two-stage로 구성된다.

## 프로젝트 내용
* TF-IDF, Dense embedding을 활용하여 질문과 관련된 지문을 추추하는 retriever를 구현한다.
* retriever가 질문의 정답이 존재하는 문서를 가져오는 성능을 평가한다.
* 질문과 문서가 input으로 주어졌을 때 정답의 위치를 예측하는 reader 모델을 구현한다.
* reader, retriever를 연결하여 open domain question answering system을 구축한다.
* 모델의 추론과 학습을 위한 preprocessing, postprocessing을 수행한다.

## 데이터셋
- Train 데이터 개수: 3952개
- Test 데이터 개수: 240개
 ![data](https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-08/assets/76895949/e50d0fe5-76a2-4f84-b412-75de6ad4ef21)

## 평가지표
- Exact Match (EM): 띄어쓰기 및 "."과 같은 문자의 존재 여부를 제외하고 정답과 예측이 정확히 일치하는 경우에 1점이 부여된다.
- F1 Score : 정답은 "Barack Obama"지만 예측이 "Obama"일 때, EM의 경우 0점을 받겠지만 F1 Score는 겹치는 단어도 있는 것을 고려해 부분 점수를 부여한다.

## 실행 방법
```
python train_reader.py --output_dir=./output_test --dataset_name=../data/train_dataset/ --model_name_or_path=klue/roberta-large --do_eval --do_train --overwrite_output_dir
```
```
python eval_reader.py --output_dir=./output_eval --dataset_name=../data/train_dataset/ --model_name_or_path=./output_test/checkpoint-3000 --overwrite_output_dir --do_eval
```
```
python inference.py --output_dir=./output_inference --dataset_name=../data/test_dataset/ --model_name_or_path=./output_epoch6/checkpoint-3000 --do_predict --overwrite_output_dir
```
```
python retrieval/train_retriever.py
python retrieval/eval_retriever.py
```

## 모델 평가 및 개선
- reader모델의 정확도 향상
  - klue/bert-base 모델을 klue/roberta-large로 교체 EM:41 -> EM:49
  - 모델 학습 옵션 조정을 통한 개선
    - 지속적으로 loss가 감소하는데 반해 EM score의 진동 폭이 굉장히 크다 overfitting 직전의 학습 환경을 세팅한다.
    - | klue/bert-base | submission | training |
      | --- | --- | --- |
      |5000 step| (submission) EM : 32.08 / f1: 44.28 | (training) loss: 0.03 / EM: 60.41 / f1 : 68.74 |
      |2500 step| (submission) EM : 34.58 / f1 : 47.65 | (training) loss: 1.35 / EM : 51 / f1: 64 |
  - 더 많은 양의 korquad 데이터셋으로 학습 후 wiki 데이터셋으로 finetuning을 수행
    | klue-bert-base | EM |
    | --- | --- |
    | korquad dataset 5000 스텝까지 학습 후 wiki로 추가 2400스텝 학습 | 40.83 |
    | wiki로만 2400 스텝 학습 | 41.67 |
    | korquad dataset 5000 스텝 학습 | 37.5 |
    -  wiki 외의 데이터 증강은 리더보드에서 도움이 되지 않는다. 
  - klue/bert-base 구조 개선
    - bert모델 위에 start/end logit을 뱉는 classification layer위에 bi-directional lstm을 올려서 추가 학습.
   
- retriever의 정확도 향상
  - query와 tf-idf score가 높은 passage를 추출하되 top-k개의 문서를 합쳐서 하나의 문서로 만든 후 reader에게 반환한다. top-k에 따른 retriever의 정확도를 확인한다.
  - | Top-k | performance | End-to-End evaluation result |
    | --- | --- | --- |
    | 1 | 1055 / 4192 (25%) | - |
    | 20 | 72% | +klue/bert-base EM41 |
    | 30 | 77% | +klue/roberta-large EM 49 |
    | 40 | 79% | - |
    | 50 | 81% | +klue/bert-base EM40 |
  - sparse embedding을 dense embedding으로 교체
    - klue/bert-base 모델을 각각 passage encoder, query encoder로 학습시킨다.
    - 질문의 정답이 있는 passage와의 embedding vector는 query vector와 가깝도록하고 랜덤으로 선택한 문서와의 embedding vector와 query의 거리는 멀게 하도록 학습시킨다.
    - korquad dataset으로 임베딩 모델을 학습시킨다.
    - wiki dataset에서 evaluation을 수행한 결과 정확도가 매우 낮아서 end-to-end eval까지 가지 못했다.
