# 챕터8 딥러닝

### **데이터 확장**
\- 정확도를 높이기 위해 입력 이미지(훈련 이미지)를 알고리즘을 동원해 '인위적'으로 확장하는 기법  
\- 입력 이미지를 회전하거나 세로로 이동하는 등 미세한 변화를 주어 이미지 개수를 늘림  
\- 데이터가 몇 개 없을 때 효과적인 수단  
\- 이미지를 일부 잘라내는 'Crop', 좌우를 뒤집는 'Flip', 밝기 등 외형 변화나 확대, 축소 등의 스케일 변화도 효과적  

## **층을 깊게 하는 이유(중요성)**
\- **신경망의 매개변수 수가 줄어듬**
- 깊은 층이 깊지 않은 층보다 적은 매개변수로 같은 수준의 표현력을 달성할 수 있음
- 매개변수는 층을 반복할수록 적어짐
- 층이 깊어질수록 그 차이는 커짐  

\- **학습의 효율성**
- 층을 깊게 함으로써 학습 데이터의 양을 줄여 학습을 고속으로 수행 가능
- 학습해야 할 문제를 계층적으로 분해 가능 (각 층이 학습해야 할 문제를 단순화)

\- **정보를 계층적으로 전달할 수 있음**

## **딥러닝의 초기 역사**
\- 큰 주목을 받게 된 계기는 이미지 인식 기술을 겨루는 장인 ILSVRC(ImageNet Large Scale Visual Recognition Challenge) 대회

### **이미지넷**
\- 100만 장이 넘는 이미지를 담고 있는 데이터셋  
\- 다양한 종류의 이미지를 포함하여 각 이미지에는 레이블이 붙어 있음  

### **VGG**
\- 합성곱 계층과 풀링 계층으로 구성되는 기본적인 CNN과 비슷  
\- 다만, 비중있는 층(합성곱 계층, 완전연결 계층)을 모두 16층 혹은 19층으로 심화한게 특징(층의 깊이에 따라 'VGG16'과 'VGG19'로 구분)  
\- 주목할 점은 3 X 3의 작은 필터를 사용한 합성곱 계층을 연속으로 거친다는 것  
\- 합성곱 계층을 2~4회 연속으로 풀링 계층을 두어 크기를 절반으로 줄임  
\- 마지막에는 완전연결 계층을 통과시켜 결과를 출력  
\- 성능에서는 GoogLeNet에 뒤지지만, 구성이 간단하여 응용하기 좋음  

### **GoogLeNet**
\- 세로 방향 깊이뿐 아니라 가로 방향도 깊다는 점이 특징  
\- 가로 방향에 '폭'이 있는데, 이를 '인셉션 구조'라고 함  
**- 인셉션 구조**
- 크기가 다른 필터(와 풀링)를 여러 개 적용하여 그 결과를 결합
- 인셉션 구조를 하나의 빌딩 블록으로 사용하는 것이 특징
- 1 X 1 크기의 필터를 사용한 합성곱 계층을 많은 곳에서 사용
- 이는 채널 쪽으로 크기를 줄이는 것으로, 매개변수 제거와 고속 처리에 기여

### **ResNet**
\- 딥러닝의 학습에서는 층이 지나치게 깊으면 학습이 잘 되지 않고, 오히려 성능이 떨어지는 경우가 많음  
\- ResNet은 이런 문제를 해결하기 위해서 "**스킵 연결**"을 도입  
**- 스킵 연결**
- 입력 데이터를 합성곱 계층을 건너뛰어 출력에 바로 더하는 구조  
- 층이 깊어져도 학습을 효율적으로 할 수 있도록 도와주는데, 이는 역전파 때 스킵 연결이 신호 감쇠를 막아주기 때문
- 핵심은 상류의 기울기에 아무런 수정을 가하지 않고 '그대로' 흘린다는 것
- 기울기가 작아지거나 지나치게 커질 걱정 없이 앞 층에 '의미 있는 기울기'가 전해지리라 기대
- 층을 깊게 할수록 기울기가 작아지는 소실 문제를 줄여줌

**- 전이 학습**
- 이미지넷이 제공하는 거대한 데이터셋으로 학습한 가중치 값들은 실제 제품에 활용해도 효과적, 이를 "**전이 학습**" 이라고 함  
- 학습된 가중치를 다른 신경망에 복사한 다음, 그 상태로 재학습을 수행
- 미리 학습된 가중치를 초깃값으로 설정한 후, 새로운 데이터셋을 대상으로 재학습을 수행
- 보유한 데이터셋이 적을 때 유용한 방법

## **딥러닝 고속화**