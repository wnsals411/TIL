# 마크다운 실습2. TIL(Today I Learned)

---

- 어제 배운 내용을 마크다운으로 구조화하여 정리 해보세요.

## Numerical Python(Numpy) - Part 1

```python
import numpy as np # 사실상 표준

a = [1, 2, 3]
a * 3
```

... [1, 2, 3, 1, 2, 3, 1, 2, 3]

```python
list(map(lambda x: 3*x, a))
```

... [3, 6, 9]

```python
b = np.array([1,2,3])
b * 3
```

... array([3, 6, 9])

### ***1. Numpy Dimensional array***

```python
a = np.array([1,4,5,8], float)
a
```

... array([1., 4., 5., 8.])

```python
type(a)
```

... numpy.ndarray

```python
a.dtype, a.shape
```

... (dtype('float64'), (4,))

### ***2. Array shape***

```python
# Vector(1차원)
vector = np.array([1,4,5,8])
vector.shape
```

... (4,)

```python
# Matrix(2차원)
matrix = np.array([[1,2,3], [4,5,16]])
print(matrix.shape)
print(matrix)
```

... (2, 3) 

​	[[ 1  2  3]

​	[ 4  5 16]]

```python
# Tensor(3차원 이상)
tensor = np.arange(1,25).reshape(2,3,4)
print(tensor)
```

...[[[ 1  2  3  4] 

​	 [ 5  6  7  8]

​	 [ 9 10 11 12]]

```python
vector.shape, matrix.shape, tensor.shape
```

... ((4,), (2, 3), (2, 3, 4))

```python
# element의 갯수
vector.size, matrix.size, tensor.size
```

... (4, 6, 24)

```python
# data type
# float32: 32bit로 표시 - single precision
# float64: 64bit로 표시 - double precision
a = np.array([1,2,3], dtype=np.float32)
a
```

... array([1., 2., 3.], dtype=float32)

```python
b = np.array([1,2,3], np.int16)
b
```

... array([1, 2, 3], dtype=int16)

reshape

```python
matrix.shape
```

... (2, 3)

```python
matrix.reshape(6,)
```

... array([ 1,  2,  3,  4,  5, 16])

```python
matrix.reshape(3,2)
```

... array([[ 1,  2],

​			  [ 3,  4],

​			  [ 5, 16]])

```python
# -1은 numpy가 알아서 맞춰줌 (단, size가 동일)
matrix.reshape(-1,)
```

... array([ 1,  2,  3,  4,  5, 16])

```python
matrix.reshape(3,-1)
```

... array([[ 1,  2],

​			  [ 3,  4],

​		  	[ 5, 16]])

```python
matrix.reshape(3,-1).shape
```

... (3, 2)

```python
matrix.reshape(-1,)
```

... array([ 1,  2,  3,  4,  5, 16])

```python
matrix.reshape(1,-1)
```

... array([[ 1,  2,  3,  4,  5, 16]])

```python
matrix.reshape(-1,1)
```

... array([[ 1],

​		      [ 2],

​		      [ 3],

​		      [ 4], 

​		      [ 5],

 	         [16]])

flatten

```python
tensor.reshape(-1,)
```

... array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

```
tensor.flatten()
```

... array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

### ***3. Indexing & Slicing***

Indexing

```python
matrix
```

... array([[ 1,  2,  3],

​		      [ 4,  5, 16]])

```python
matrix[0][1]
```

... 2

```python
matrix[0,1]
```

... 2

```python
tensor[1,1,2]
```

... 19

slicing

```python
b = np.arange(16).reshape(4,-1)
b
```

... array([[ 0,  1,  2,  3],

​		      [ 4,  5,  6,  7], 

​		      [ 8,  9, 10, 11],

​			  [12, 13, 14, 15]])

```python
# 5,6,9,10
b[1:3, 1:3]
```

... array([[ 5,  6],

 	         [ 9, 10]])

```python
# 5,6,7, 9,10,11
b[1:3, 1:]
```

... array([[ 5,  6,  7],

​	         [ 9, 10, 11]])

```python
# 1,3,9,11
b[::2, 1::2]
```

... array([[ 1,  3],

  	        [ 9, 11]])

```python
# X는 앞에 3열, y는 마지막 열
X, y = b[:, :-1], b[:, -1]
X.shape, y.shape
```

... ((4, 3), (4,))