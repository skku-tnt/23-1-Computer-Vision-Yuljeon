# ResNet

Status: Study

# Deep Residual Learning for Image Recognition

## Abstract

- Neural network의 깊이가 깊어질수록 학습이 어려워짐
- 이를 해결하기 위해 residual learning framework를 제시
- layer에 입력되는 input 값 고려하는 residual function 사용
- 최적화가 용이하고 깊이가 깊어질수록 accuracy가 높아지는 장점
- 다양한 visual recognition task에 활용됨

## ResNet의 등장 배경

- 레이어를 많이 쌓아도 학습 성능이 높아지지 않는 문제 발생
- 레이어의 크기를 키운 모델들의 파라미터 개수가 너무 많다는 문제 발생
    1. vanishing/exploding gradients 
    - sigmoid 같은 activation func 사용 시 레이어가 깊어지면 역전파 과정에서 gradient가 0에 가까워져 학습이 어려워지는 경우
    - convergence 방해
    
    → normalized initialization (가중치 초기화)로 문제 해결
    
    1. **degradation problem** 
        
        ![Untitled](ResNet%2038adff022e9441179fc3b7aecd1b979a/Untitled.png)
        
    - 네트워크 깊이가 깊어지면 accuracy가 saturated (가중치 업데이트 중단 현상)
    - training error 높아짐
    - overfitting 때문 X
    - 시스템마다 최적화 방식이 동일하지 않다는 점을 의미
    
    → **추가되는 레이어들은 identity mapping으로, 나머지 레이어들은 얕은 모델에서 학습된 내용 사용**
    

→ **Residual block**을 활용하여 네트워크의 optimization 난이도 (모델 학습 전반의 난이도)를 낮추자는 개념 등장

## Deep Residual Learning

![Untitled](ResNet%2038adff022e9441179fc3b7aecd1b979a/Untitled%201.png)

![Untitled](ResNet%2038adff022e9441179fc3b7aecd1b979a/Untitled%202.png)

기존 학습(왼쪽)에서 각 Layer를 단순히 학습시켜왔다면, Residual Learning(오른쪽)에서는 새롭게 추가된 내용만 학습시킨 후 기존 내용을 더해주는 형식으로 간단화 함

- H(x): 궁극적으로 학습하고자 하는 내용
- F(x): 새로 학습하고자 하는 내용
- x: 기존에 학습된 내용

- F(x) = H(x) - x
- H(x) = F(x) + x

→ F(x) 학습 후 x를 더해주는 **shortcut connections, identity mapping** 의 장점

- 파라미터 추가 X
- 연산 복잡성 X
- 레이어 깊어질수록 성능 향상

## Identity Mapping by Shortcuts

![Untitled](ResNet%2038adff022e9441179fc3b7aecd1b979a/Untitled%203.png)

- F: 새롭게 학습되어야 하는 내용
- x: 기존에 학습된 내용
- y: output

- **F와 x의 차원이 동일해야 함 → identity mapping 수행**
- 차원이 다르다면? x에 Ws를 곱해 **linear projection** 수행
- 레이어가 단 1개이면 이득이라고 보긴 어려움

## Network Architectures

- Plain network

![Untitled](ResNet%2038adff022e9441179fc3b7aecd1b979a/Untitled%204.png)

- feature map의 사이즈와 filter 개수 동일
- feature map의 사이즈가 절반 작아지면 filter 개수는 2배
- 왼쪽의 VGG-19와 비교할 때 훨씬 낮은 계산 복잡도 (FLOPs)

- Residual network

![Untitled](ResNet%2038adff022e9441179fc3b7aecd1b979a/Untitled%205.png)

- Plain network에 shortcut connections을 추가한 구조
- 실선: input과 output의 차원 동일 → identity shortcuts
- 점선: input과 output의 차원 다름 (차원 증가)

→ Option A: identity mapping 수행 + 증가한 차원에 대해 zero padding

→ Option B: projection shortcut 수행 (1x1 convolutions)

(두 옵션 모두 dimension이 달라질 땐 stride를 2로 설정하여 너비와 높이 줄임)

## Experiments

- ImageNet 대상 모델 구조

![Untitled](ResNet%2038adff022e9441179fc3b7aecd1b979a/Untitled%206.png)

(50-layer부터는 layer가 3개씩 구성되는 bottleneck building block 사용)

- Residual Networks의 주요 장점
    - training error 낮음
    - degradation problem 해결
    - accuracy 향상
    - faster convergence를 통한 optimization 용이

- Bottleneck Architecture
    - 1 x 1 convolution layer를 거치면서 차원을 축소하고 계산 연산량을 줄일 수 있으나 정보 손실의 문제가 있음 (trade-off)

- identity vs projection shortcuts
    - 성능의 큰 격차 없음
    - bottleneck architectures (레이어가 3개)의 복잡성을 낮추기 위해 identity shortcut 자주 사용

- ImageNet뿐만 아니라 CIFAR-10 등에 대해서도 좋은 성능 보임

![왼: plain, 오: ResNet](ResNet%2038adff022e9441179fc3b7aecd1b979a/Untitled%207.png)

왼: plain, 오: ResNet

### 질문

- convergence의 의미, 역할?
    - 경사하강법에서 gradient를 줄여나가는 것
    - 0 쪽으로 수렴
- stochastic gradient descent (확률적 경사 하강법)
    - 확률적으로 gradient를 찾아가는 방식
    - 이 당시에 많이 쓰인 방식
- identity mapping == shortcut connection
    - 기존 Input 값을 새로 학습한 내용에 더해주는 것