# Q.11 - 20

## Q.11. 均值滤波器

使用均值滤波器（3x3）来进行滤波吧！

均值滤波器使用网格内像素的平均值。

| 输入 (../assets/imori.jpg) | 输出 (img/answer_11.jpg) |
| :--------------: | :--------------------------: |
|  ![](../assets/imori.jpg)  |  ![](img/answer_11.jpg)  |




## Q.12. Motion Filter

使用3x3的 Motion Filter 来进行滤波吧。

Motion Filter 取对角线方向的像素的平均值，像下式这样定义：

```bash
  1/3  0   0
[  0  1/3  0 ]
   0   0  1/3
```

| 输入 (../assets/imori.jpg) | 输出 (img/answer_12.jpg) |
| :--------------: | :--------------------------: |
|  ![](../assets/imori.jpg)  |  ![](img/answer_12.jpg)  |



## Q.13. MAX-MIN 滤波器

使用 MAX-MIN 滤波器来进行滤波吧。

MAX-MIN 滤波器使用网格内像素的最大值和最小值的差值对网格内像素重新赋值。通常用于**边缘检测**。边缘检测用于检测图像中的线。像这样提取图像中的信息的操作被称为**特征提取**。边缘检测通常在灰度图像上进行。

| 输入 (../assets/imori.jpg) | 输出 (img/answer_13.jpg) |
| :--------------: | :--------------------------: |
|  ![](../assets/imori.jpg)  |  ![](img/answer_13.jpg)  |




## Q.14. 微分滤波器

使用3x3的微分滤波器来进行滤波吧。

微分滤波器对图像亮度急剧变化的边缘有提取效果，可以获得邻接像素的差值。

```bash
    (a)纵向         (b)横向
      0 -1  0            0 0 0
K = [ 0  1  0 ]   K = [ -1 1 0 ]
      0  0  0            0 0 0
```

| 输入 (../assets/imori.jpg) | 输出・纵向 (img/answer_14_v.jpg) | 输出・横向 (img/answer_14_h.jpg) |
| :--------------: | :----------------------------------: | :----------------------------------: |
|  ![](../assets/imori.jpg)  |     ![](img/answer_14_v.jpg)     |     ![](img/answer_14_h.jpg)     |



## Q.15. Sobel 滤波器

使用3x3的 Sobel 滤波器来进行滤波吧。

Sobel 滤波器可以提取特定方向的边缘，滤波器按下式定义：

```bash
    (a)纵向       (b)横向
      1 0 -1            1  2  1
K = [ 2 0 -2 ]   K = [  0  0  0 ]
      1 0 -1           -1 -2 -1
```

| 输入 (../assets/imori.jpg) | 输出・纵向 (img/answer_15_v.jpg) | 输出・横向 (img/answer_15_h.jpg) |
| :--------------: | :----------------------------------: | :----------------------------------: |
|  ![](../assets/imori.jpg)  |     ![](img/answer_15_v.jpg)     |     ![](img/answer_15_h.jpg)     |



## Q.16. Prewitt 滤波器

使用3x3的 Prewitt 滤波器来进行滤波吧。

Prewitt 滤波器是用于边缘检测的一种滤波器，使用下式定义：

```bash
    (a)纵向          (b)横向
      -1 -1 -1          -1 0 1
K = [  0  0  0 ]  K = [ -1 0 1 ]
       1  1  1          -1 0 1
```

| 输入 (../assets/imori.jpg) | 输出・纵向 (img/answer_16_v.jpg) | 输出・横向 (img/answer_16_h.jpg) |
| :--------------: | :----------------------------------: | :----------------------------------: |
|  ![](../assets/imori.jpg)  |     ![](img/answer_16_v.jpg)     |     ![](img/answer_16_h.jpg)     |




## Q.17. Laplacian 滤波器

使用 Laplacian 滤波器来进行滤波吧。

Laplacian 滤波器是对图像亮度进行二次微分从而检测边缘的滤波器。由于数字图像是离散的，x 方向和 y 方向的一次微分分别按照以下式子计算：

```bash
Ix(x,y) = (I(x+1, y) - I(x,y)) / ((x+1)-x) = I(x+1, y) - I(x,y)
Iy(x,y) = (I(x, y+1) - I(x,y)) / ((y+1)-y) = I(x, y+1) - I(x,y)
```

因此二次微分按照以下式子计算：

```bash
Ixx(x,y) = (Ix(x,y) - Ix(x-1,y)) / ((x+1)-x) = Ix(x,y) - Ix(x-1,y)
         = (I(x+1, y) - I(x,y)) - (I(x, y) - I(x-1,y))
         = I(x+1,y) - 2 * I(x,y) + I(x-1,y)
Iyy(x,y) = ... = I(x,y+1) - 2 * I(x,y) + I(x,y-1)
```

特此，Laplacian 表达式如下：

```bash
D^2 I(x,y) = Ixx(x,y) + Iyy(x,y)
           = I(x-1,y) + I(x,y-1) - 4 * I(x,y) + I(x+1,y) + I(x,y+1)
```

如果把这个式子表示为卷积核是下面这样的：

```bash
      0  1  0
K = [ 1 -4  1 ]
      0  1  0
```

| 输入 (../assets/imori.jpg) | 输出(img/answer_17.jpg) |
| :--------------: | :-------------------------: |
|  ![](../assets/imori.jpg)  | ![](img/answer_17.jpg)  |  |



## Q.18. Emboss 滤波器

使用 Emboss 滤波器来进行滤波吧。

Emboss 滤波器可以使物体轮廓更加清晰，按照以下式子定义：

```bash
      -2 -1  0
K = [ -1  1  1 ]
       0  1  2
```

| 输入 (../assets/imori.jpg) | 输出(img/answer_18.jpg) |
| :--------------: | :-------------------------: |
|  ![](../assets/imori.jpg)  | ![](img/answer_18.jpg)  |



## Q.19. LoG 滤波器

使用 LoG  滤波器，来对`../assets/imori_noise.jpg`检测边缘吧！

 LoG  即高斯-拉普拉斯（ Laplacian of Gaussian ）的缩写，使用高斯滤波器使图像平滑化之后再使用拉普拉斯滤波器使图像的轮廓更加清晰。

为了防止拉普拉斯滤波器计算二次微分会使得图像噪声更加明显，所以我们首先使用高斯滤波器来抑制噪声。

 LoG  滤波器使用以下式子定义：

```bash
LoG(x,y) = (x^2 + y^2 - s^2) / (2 * pi * s^6) * exp(-(x^2+y^2) / (2*s^2))
```

| 输入 (../assets/imori_noise.jpg) | 输出 (img/answer_19.jpg) |
| :--------------------: | :--------------------------: |
|  ![](../assets/imori_noise.jpg)  |  ![](img/answer_19.jpg)  |



## Q.20. 直方图

使用`Matplotlib`来绘制`../assets/imori_dark.jpg`的直方图吧！

直方图显示了不同数值的像素出现的次数。在`Matplotlib`中有`hist()`函数提供绘制直方图的接口。

| 输入 (../assets/imori_dark.jpg) | 输出 (img/answer_20.png) |
| :-------------------: | :--------------------------: |
|  ![](../assets/imori_dark.jpg)  |  ![](img/answer_20.png)  |



