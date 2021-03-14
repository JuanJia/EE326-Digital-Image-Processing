## Python DIP PIL中 convert('L') 函数原理

### img = img.convert()

PIL有九种不同模式: 1，L，P，RGB，RGBA，CMYK，YCbCr，I，F。

![origin](E:\OneDrive\学习\课程\大三春季学期\数字图像处理\Lab\lab3\note\origin.png)

### 1. img.convert('1')

为二值图像，非黑即白。每个像素用8个bit表示，0表示黑，255表示白。

```python
1 from PIL import Image
2 
3 
4 def convert_1():
5     image = Image.open("origin.jpg")
6     image_1 = image.convert('1')
7     image.show()
8     image_1.show()
```

![origin](E:\OneDrive\学习\课程\大三春季学期\数字图像处理\Lab\lab3\note\1.png)

### 2. img.convert('L')

为灰度图像，每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。转换公式：L = R * 299/1000 + G * 587/1000+ B * 114/1000。

```python
1 from PIL import Image
2 
3 
4 def convert_L():
5     image = Image.open("origin.jpg")
6     image_L = image.convert('L')
7     image.show()
8     image_L.show()
```

![origin](E:\OneDrive\学习\课程\大三春季学期\数字图像处理\Lab\lab3\note\L.png)

### 1.3 img.convert('P')

```python
from PIL import Image
2 
3 
4 def convert_P():
5     image = Image.open("origin.jpg")
6     image_P = image.convert('P')
7     image.show()
8     image_P.show()
```