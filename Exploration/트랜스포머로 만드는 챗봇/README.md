# Code Peer Review Templete
- 코더 : 김재림
- 리뷰어 : 신노아

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

- 트랜스포머 모델 구현이 훌륭하게 되어 안정적인 수렴을 보여준 것이 인상적입니다. 
- 코드 흐름이 자연스럽고 주석 설명이 알기 쉬워서 많은 걸 배웠습니다. 
- 한국어 챗봇학습이 잘 이루어졌으며 결과도 우수합니다.


```python
# 사칙 연산 계산기
class calculator:
    # 예) init의 역할과 각 매서드의 의미를 서술
    def __init__(self, first, second):
        self.first = first
        self.second = second
    
    # 예) 덧셈과 연산 작동 방식에 대한 서술
    def add(self):
        result = self.first + self.second
        return result

a = float(input('첫번째 값을 입력하세요.')) 
b = float(input('두번째 값을 입력하세요.')) 
c = calculator(a, b)
print('덧셈', c.add()) 
```
