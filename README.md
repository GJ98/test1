# Format

## dataset
```JSON
{
    "id": str,
    "title": str,
    "context": str,
    "question": str,
    "answers": {
        "text": List[str],
        "answer_start": List[int]
    }
}
```
## model output
```JSON
# MRC
{
    "predictions": [{
        "id": str,
        "prediction_text": str
    }], //len=(# of data point)
    "label_ids": [{
        "id": str,
        "answers": {
            "text": List[str],
            "answer_start": List[int]
        }
    }] //len=(# of data point)
}

# Retrieval
{
    "sim_score": np.array, //shape=(# of data point, num_neg+1)
    "targets": np.array //shape=(# of data point)
}
```

# configuration
## MRC
- [x] 데이터 전처리: 변경 가능 (modify `pl.datamodule.setup` in `/module/data.py`)
- [ ] ~모델링: 변경 불가 (only use `AutoModelForQuestionAnswering`)~
- [ ] ~손실 함수: 변경 불가 (only use `AutoModelForQuestionAnswering`)~
- [x] 최적화 함수: 변경 가능 (if exists in `torch.optim`)
- [x] 평가 지표: 변경 가능 (if exsits in `datasets_modules.metrics`) 이떄, 입력 구조를 수정해야 한다.

## Retrieval
- [x] 데이터 전처리: 변경 가능 (modify `pl.datamodule.setup` in `/module/data.py`)
- [x] 모델링: 변경 가능 (custom model class in `/module/encoder`)
- [x] 손실 함수: 변경 가능 (custom loss function in `/module/loss`)
- [x] 최적화 함수: 변경 가능 (if exists in `torch.optim`)
- [x] 평가 지표: 변경 가능 (if exsits in `datasets_modules.metrics`)

# TODO
- [ ] `retriever`의 `create_embedding_vector`에서 원하는 문서를 선택해 matrix를 구성하게 만들기
- [ ] 추론 시 retriever와 mrc를 연결하는 파이프라인 구현하기
- [x] `log` 설정하기
- [x] `wandb` 연결하기
- [x] `sweep` 설정 구현하기
- [ ] output 형식 맞추기


validation에 loss 추가하기