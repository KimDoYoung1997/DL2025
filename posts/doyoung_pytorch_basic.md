---
title: "01wk-1: (토치) -- 강의소개, 파이토치 기본"
author: "최규빈"
date: "03/05/2025"
---
<a href="https://colab.research.google.com/github/guebin/DL2025/blob/main/posts/01wk-1.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="text-align: left"></a>

# 1. 강의영상

{{<video https://youtu.be/playlist?list=PLQqh36zP38-w2wjg7vQJHPP89wWGSq-MZ&si=J0lxSPEePOD-5BOL >}}

# 2. Imports


```python
import torch
```

# 3. 환경셋팅 

`-` 코랩이용자: 별도의 설치 필요 없음 

- GPU 관리가 꼭 필요함!! (시험하루전날 수업 몰아서 듣지않기, 아이디만들어두기)
- 특히 시험기간에 조심하세요 

`-` 리눅스서버 사용자 (ref: <https://pytorch.org/get-started/locally/>)

```bash
conda create -n dl2025 python=3.9
conda activate dl2025 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

# 4. 필요한 지식 

`-` 선형대수학

- 벡터와 행렬
- 행렬의 곱셉
- 트랜스포즈

`-` 기초통계학(수리통계)

- 정규분포, 이항분포
- 모수, 추정
- $X_i \overset{i.i.d.}{\sim} N(0,1)$


`-` 회귀분석 

- 반응변수($y$), 설명변수($X$)
- ${\boldsymbol y} = {\bf X}{\boldsymbol \beta} + {\boldsymbol \epsilon}$

`-` 파이썬

- 파이썬 기본문법
- 넘파이, 판다스
- 전반적인 클래스 지식 (`__init__`, `self`, ...)
- 상속

# 5. 파이토치 기본

## A. torch 

`-` 벡터


```python
torch.tensor([1,2,3])
```




    tensor([1, 2, 3])



`-` 벡터의 덧셈


```python
torch.tensor([1,2,3]) + torch.tensor([2,2,2])
```




    tensor([3, 4, 5])



`-` 브로드캐스팅 


```python
torch.tensor([1,2,3]) + 2
```




    tensor([3, 4, 5])



## B. 벡터와 매트릭스 

`-` $3 \times 2$ matrix


```python
torch.tensor([[1,2],[3,4],[5,6]]) 
```




    tensor([[1, 2],
            [3, 4],
            [5, 6]])



`-` $3 \times 1$ matrix = $3 \times 1$ column vector


```python
torch.tensor([[1],[3],[5]]) 
```




    tensor([[1],
            [3],
            [5]])



`-` $1 \times 2$ matrix = $1 \times 2$ row vector


```python
torch.tensor([[1,2]]) 
```




    tensor([[1, 2]])



`-` 더하기 

***브로드캐스팅(편한거)***


```python
torch.tensor([[1,2],[3,4],[5,6]]) - 1
```




    tensor([[0, 1],
            [2, 3],
            [4, 5]])




```python
torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([[-1],[-3],[-5]])
```




    tensor([[0, 1],
            [0, 1],
            [0, 1]])




```python
torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([[-1,-2]])
```




    tensor([[0, 0],
            [2, 2],
            [4, 4]])



***잘못된 브로드캐스팅***


```python
torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([[-1,-3,-5]])
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[11], line 1
    ----> 1 torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([[-1,-3,-5]])


    RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1



```python
torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([[-1],[-2]])
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[12], line 1
    ----> 1 torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([[-1],[-2]])


    RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0


***이상한 것***


```python
torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([-1,-2])
```




    tensor([[0, 0],
            [2, 2],
            [4, 4]])




```python
torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([-1,-3,-5])
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[14], line 1
    ----> 1 torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([-1,-3,-5])


    RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1


`-` 행렬곱

***정상적인 행렬곱***


```python
torch.tensor([[1,2],[3,4],[5,6]]) @ torch.tensor([[1],[2]])
```




    tensor([[ 5],
            [11],
            [17]])




```python
torch.tensor([[1,2,3]]) @ torch.tensor([[1,2],[3,4],[5,6]]) 
```




    tensor([[22, 28]])



***잘못된 행렬곱***


```python
torch.tensor([[1,2],[3,4],[5,6]]) @ torch.tensor([[1,2]])
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[17], line 1
    ----> 1 torch.tensor([[1,2],[3,4],[5,6]]) @ torch.tensor([[1,2]])


    RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x2 and 1x2)



```python
torch.tensor([[1],[2],[3]]) @ torch.tensor([[1,2],[3,4],[5,6]]) 
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[18], line 1
    ----> 1 torch.tensor([[1],[2],[3]]) @ torch.tensor([[1,2],[3,4],[5,6]]) 


    RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x1 and 3x2)


***이상한 것***


```python
torch.tensor([[1,2],[3,4],[5,6]]) @ torch.tensor([1,2]) # 이게 왜 가능..
```




    tensor([ 5, 11, 17])




```python
torch.tensor([1,2,3]) @ torch.tensor([[1,2],[3,4],[5,6]]) # 이건 왜 가능?
```




    tensor([22, 28])



## C. transpose, reshape

`-` transpose 


```python
torch.tensor([[1,2],[3,4]]).T 
```




    tensor([[1, 3],
            [2, 4]])




```python
torch.tensor([[1],[3]]).T 
```




    tensor([[1, 3]])




```python
torch.tensor([[1,2]]).T 
```




    tensor([[1],
            [2]])



`-` reshape

***일반적인 사용***


```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(2,3)
```




    tensor([[1, 2, 3],
            [4, 5, 6]])




```python
torch.tensor([[1,2],[3,4],[5,6]])
```




    tensor([[1, 2],
            [3, 4],
            [5, 6]])




```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(1,6)
```




    tensor([[1, 2, 3, 4, 5, 6]])




```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(6)
```




    tensor([1, 2, 3, 4, 5, 6])



***편한 것***


```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(2,-1)
```




    tensor([[1, 2, 3],
            [4, 5, 6]])




```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(6,-1)
```




    tensor([[1],
            [2],
            [3],
            [4],
            [5],
            [6]])




```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(-1,6)
```




    tensor([[1, 2, 3, 4, 5, 6]])




```python
torch.tensor([[1,2],[3,4],[5,6]]).reshape(-1)
```




    tensor([1, 2, 3, 4, 5, 6])



## D. concat, stack $(\star\star\star)$

`-` concat 


```python
a = torch.tensor([[1],[3],[5]])
b = torch.tensor([[2],[4],[6]])
print(a.shape)
print(b.shape)
torch.concat([a,b],axis=1)
```

    torch.Size([3, 1])
    torch.Size([3, 1])





    tensor([[1, 2],
            [3, 4],
            [5, 6]])




```python
torch.concat([a,b])
```




    tensor([[1],
            [3],
            [5],
            [2],
            [4],
            [6]])




```python
torch.concat([a,b],axis=0)
```




    tensor([[1],
            [3],
            [5],
            [2],
            [4],
            [6]])




```python
torch.concat([a,b],axis=1)
```




    tensor([[1, 2],
            [3, 4],
            [5, 6]])




```python
torch.concat([a,b],axis=2)
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    Cell In[36], line 1
    ----> 1 torch.concat([a,b],axis=2)


    IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)



```python
aa = torch.tensor([[1,2],[3,4],[5,6]])  # 3x2   
bb = torch.tensor([[7,8],[9,0],[10,2]]) # 3x2
torch.concat([aa,bb])   # 6x2
```




    tensor([[ 1,  2],
            [ 3,  4],
            [ 5,  6],
            [ 7,  8],
            [ 9,  0],
            [10,  2]])




```python
torch.concat([aa,bb]).shape
```




    torch.Size([6, 2])




```python
print(torch.concat([aa,bb],axis=0).shape)
print(torch.concat([aa,bb],axis=1).shape)
print(torch.concat([aa,bb],axis=2).shape)
```

    torch.Size([6, 2])
    torch.Size([3, 4])



    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    Cell In[39], line 3
          1 print(torch.concat([aa,bb],axis=0).shape)
          2 print(torch.concat([aa,bb],axis=1).shape)
    ----> 3 print(torch.concat([aa,bb],axis=2).shape)


    IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)



```python
aaa = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]) # 2 x 2 x 3
bbb = torch.tensor([[[13,14,15],[16,17,18]],[[19,20,21],[22,23,24]]]) # 2 x 2 x 3
print(aaa.shape)
print(bbb.shape)
```

    torch.Size([2, 2, 3])
    torch.Size([2, 2, 3])



```python
print(torch.concat([aaa,bbb],axis=0).shape)
print(torch.concat([aaa,bbb],axis=1).shape)
print(torch.concat([aaa,bbb],axis=2).shape)
```

    torch.Size([4, 2, 3])
    torch.Size([2, 4, 3])
    torch.Size([2, 2, 6])


`-` stack


```python
a = torch.tensor([1,3,5])
b = torch.tensor([2,4,6])
print(a.shape)
print(b.shape)
torch.stack([a,b],axis=1)
```

    torch.Size([3])
    torch.Size([3])





    tensor([[1, 2],
            [3, 4],
            [5, 6]])




```python
torch.concat([a.reshape(3,1),b.reshape(3,1)],axis=1)
```




    tensor([[1, 2],
            [3, 4],
            [5, 6]])



note: np.concatenate은 축의 총 갯수를 유지하면서 결합, np.stack은 축의 갯수를 하나 증가시키면서 결합



:::{.callout-warning}

concat과 stack을 지금 처음본다면 아래를 복습하시는게 좋습니다. 

<https://guebin.github.io/PP2024/posts/06wk-2.html#numpy와-축axis>
:::
