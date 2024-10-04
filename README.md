# MRC
- [ ] 데이터 전처리: 변경 가능 (modify `pl.datamodule.setup` in `/module/data.py`)
- [ ] ~모델링: 변경 불가 (only use `AutoModelForQuestionAnswering`)~
- [ ] ~손실 함수: 변경 불가 (only use `AutoModelForQuestionAnswering`)~
- [x] 최적화 함수: 변경 가능 (if exists in `torch.optim`)
- [x] 평가 지표: 변경 가능 (if exsits in `datasets_modules.metrics`) 이떄, 입력 구조를 수정해야 한다.

# Retrieval
- [ ] 데이터 전처리: 변경 가능 (modify `pl.datamodule.setup` in `/module/data.py`)
- [ ] 모델링: 변경 가능 (custom model class in `/module/encoder`)
- [x] 손실 함수: 변경 가능 (custom loss function in `/module/loss`)
- [x] 최적화 함수: 변경 가능 (if exists in `torch.optim`)
- [ ] 평가 지표: 변경 가능 (if exsits in `datasets_modules.metrics`)

내일 이거 config로 설정하기