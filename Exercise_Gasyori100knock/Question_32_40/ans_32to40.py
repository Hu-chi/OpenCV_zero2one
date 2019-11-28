import cv2
import numpy as np


def fourier_transform(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.fft.fft2(img)
    # other version:
    # img_f = np.zeros_like(img, dtype=np.complex)
    # height, width = img.shape[:2]
    # x = np.tile(np.arange(width), (height, 1))
    # y = np.arange(height).repeat(width).reshape(height, -1)
    # for l in range(height):
    #     for k in range(width):
    #         img_f[l, k] = np.sum(img * np.exp(-2j * np.pi * (x * k / width + y * l / height)))
    # img = img_f
    return img


def inv_fourier_transform(img: np.ndarray) -> np.ndarray:
    img = np.fft.ifft2(img)
    # other version
    # img_if = np.zeros_like(img, dtype=np.complex)
    # height, width = img.shape[:2]
    # x = np.tile(np.arange(width), (height, 1))
    # y = np.arange(height).repeat(width).reshape(height, -1)
    # for l in range(height):
    #     for k in range(width):
    #         img_if[l, k] = np.sum(img * np.exp(2j * np.pi * (x * k / width + y * l / height))) / (height * width)
    # img = img_if
    img = np.clip(img.real, 0, 255)
    return img.astype(np.uint8)


def pass_filter(img: np.ndarray, pass_type: str = 'low', threshold1: float = 0.1, threshold2: float = 1) -> np.ndarray:
    pass_type = pass_type.lower()
    img = fourier_transform(img)
    img = np.fft.fftshift(img)
    height, width = img.shape[:2]
    x = np.tile(np.arange(width), (height, 1))
    y = np.arange(height).repeat(width).reshape(height, -1)
    dist_t = np.sqrt((x - width // 2) ** 2 + (y - height // 2) ** 2)
    mask = np.ones_like(img, dtype=np.float)
    if pass_type == 'low':
        mask[dist_t > (width // 2 * threshold1)] = 0
    elif pass_type == 'high':
        mask[dist_t < (width // 2 * threshold1)] = 0
    elif pass_type == 'band':
        if threshold1 > threshold2:
            threshold1, threshold2 = threshold2, threshold1
        mask[dist_t < (width // 2 * threshold1)] = 0
        mask[dist_t > (width // 2 * threshold2)] = 0
    else:
        print("pass type wrong! Only support low/high pass filter")
    img *= mask
    img = np.fft.ifftshift(img)
    return inv_fourier_transform(img)


def discrete_cos_transform(img: np.ndarray) -> np.ndarray:
    # TODO(huchi): extend shape without height == with
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.float32(img)
    img_dct = cv2.dct(img)

    # other version
    # height, width = img.shape[:2]
    # assert height == width
    # a = np.zeros_like(img)
    # a[0, :] = 1 / np.sqrt(2)
    # for i in range(1, height):
    #     for j in range(width):
    #         a[i, j] = np.cos(np.pi * i * (2*j + 1) / (2 * height))
    # a /= np.sqrt(height / 2.)
    # img_dct = np.matmul(a, img)
    # img_dct = np.matmul(img_dct, a.T)

    return img_dct


def inv_discrete_cos_transform(img: np.ndarray) -> np.ndarray:
    img_idct = cv2.idct(img)

    # other version
    # height, width = img.shape[:2]
    # assert height == width
    # a = np.zeros_like(img)
    # a[0, :] = 1 / np.sqrt(2)
    # for i in range(1, height):
    #     for j in range(width):
    #         a[i, j] = np.cos(np.pi * i * (2*j + 1) / (2 * height))
    # a /= np.sqrt(height / 2.)
    # img_idct = np.matmul(a.T, img)
    # img_idct = np.matmul(img_idct, a)

    img_idct = np.clip(img_idct.real, 0, 255)
    return img_idct.astype(np.uint8)


def img_zip(img: np.ndarray) -> np.ndarray:
    # TODO(huchi): learn and implement the function
    pass


my_function_map = {
    "method32": lambda x: inv_fourier_transform(fourier_transform(x)),
    "method33": lambda x: pass_filter(x, 'low', 0.5),
    "method34": lambda x: pass_filter(x, 'high', 0.2),
    "method35": lambda x: pass_filter(x, 'band', 0.1, 0.5),
    "method36": lambda x: inv_discrete_cos_transform(discrete_cos_transform(x)),
    # "method37" seems nothing to do

}


def test_function():
    img = cv2.imread("../assets/imori.jpg")
    img_ = my_function_map['method36'](img)
    cv2.imshow("result", img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_function()
