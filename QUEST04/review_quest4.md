# Code Peer Review Templete
- 코더 :  김재림
- 리뷰어 : 차정은


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요.
- [x] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- [x] 주석을 보고 작성자의 코드가 이해되었나요?
- [x] 코드가 에러를 유발할 가능성이 있나요?
- [x] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
- [x] 코드가 간결한가요?

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
import random
import time

## Fish 클래스 생성, name과 speed 속성 initialize
class Fish:
  def __init__(self, name, speed):
    self.name = name
    self.speed = speed

## 컴프리헨션 활용해서 움직임 출력하는 함수 만들기
def show_fish_movement_comprehension(fish_list):
  [print(f"{fish.name} is swimming at {fish.speed} m/s") for fish in fish_list]

print("Using Comprehension:")
show_fish_movement_comprehension(fish_list)


## 이터레이터 활용해서 움직임 출력하는 함수 만들기
def show_fish_movement_iterator(fish_list):
    for fish in fish_list:
      print(f"{fish.name} is swimming at {fish.speed} m/s")

print("Using Iterator: ")
show_fish_movement_iterator(fish_list)


## 제너레이터 활용해서 움직임 출력하는 함수 만들기
def show_fish_movement_generator(fish_list):
    for fish in fish_list:
        yield f"{fish.name} is swimming at {fish.speed} m/s"
# 이 부분은 실제 출력 성공까지 완성하지 못해서 아쉬웠습니다. 

# 출력이 완성 되어서 결과값이 제대로 나왔습니다! 완성하는게 제일 중요하죠 ㅎㅎ
# f"{fish.name} is swimming at {fish.speed} m/s" 양식이 출력 함수마다 매번 들어가니까 클래스에 메서드로 넣어주셨으면 더 깔끔한 코드가 되었을 것 같습니다! 
# 제너레이터 활용한 부분은 저도 완전히 성공하진 못해서 같이 고민하면서 개선하면 좋을 것 같습니다 :)

```

# 참고 링크 및 코드 개선 여부
```python
#
#
#
#
```