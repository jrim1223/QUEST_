# Code Peer Review Templete

- 코더 : 김재림 님
- 리뷰어 : 박동원

---

# PRT(PeerReviewTemplate)

각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요.

- [x] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- [x] 주석을 보고 작성자의 코드가 이해되었나요?
- [ ] 코드가 에러를 유발할 가능성이 있나요?
- [x] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
- [x] 코드가 간결한가요?

---

# 참고 링크 및 코드 개선 여부

- 모델완성이 잘 되 었고 학습 까지 잘 이루어 진걸 확인 했습니다
- 인터뷰 결과 내용에 대해 정말 잘 알고 계셔서 잘 배우고 갑니다
- 마지막에 디코더 에대해 정의가 되지 않았다는 에러를 확인 했습니다.
- 마지막 부분 조금만 손보시면 모델의 요약내용과 실제 요약 내용을 비교 하실수 있을 것 같습니다.
- 코드가 간결하고 알아 보기 쉬워 좋았습니다

아래는 오류 가 있는 코드 내용 입니다.


```python

# 원문의 정수 시퀀스를 텍스트 시퀀스로 변환
def seq2text(input_seq):
    temp=''
    for i in input_seq:
        if (i!=0):
            temp = temp + src_index_to_word[i]+' '
    return temp

# 요약문의 정수 시퀀스를 텍스트 시퀀스로 변환
def seq2summary(input_seq):
    temp = ''
    for i in input_seq:
        if (i == tar_word_to_index['eostoken']):
            break
        if (i not in [tar_word_to_index['sostoken'], tar_word_to_index['eostoken'], 0]):
            temp = temp + tar_index_to_word[i] + ' '
    return temp        ---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-158-530b1b998bf7> in <cell line: 1>()
      2     print("원문 :", seq2text(encoder_input_test[i]))
      3     print("실제 요약 :", seq2summary(decoder_input_test[i]))
 
```


이상 입니다.
