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
    # TODO(huchi): learn and implement the function
    pass


def img_zip(img: np.ndarray) -> np.ndarray:
    # TODO(huchi): learn and implement the function
    pass


my_function_map = {
    "method32": lambda x: inv_fourier_transform(fourier_transform(x)),
    "method33": lambda x: pass_filter(x, 'low', 0.5),
    "method34": lambda x: pass_filter(x, 'high', 0.2),
    "method35": lambda x: pass_filter(x, 'band', 0.1, 0.5)
}


def test_function():
    img = cv2.imread("../assets/imori.jpg")
    img_ = my_function_map['method35'](img)
    cv2.imshow("result", img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_function()
