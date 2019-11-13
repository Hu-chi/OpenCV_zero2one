import cv2
import numpy as np
from matplotlib import pyplot as plt


def histogram_normalization(img: np.ndarray, limit_l: int = 0, limit_r: int = 255) -> np.ndarray:
    if (limit_r < limit_l):
        limit_l, limit_r = limit_r, limit_l
    height, width = img.shape[:2]
    channel = None
    if len(img.shape) < 3:
        channel = img.shape[2]
    img_min = img.min()
    img_max = img.max()
    if img_min == img_max:
        return np.full_like(img, limit_l + limit_r >> 1)

    final_img = img.copy()
    final_img[final_img < limit_l] = limit_l
    final_img[final_img > limit_r] = limit_r
    final_img = (limit_r - limit_l) / (img_max - img_min) * (final_img - img_min) + limit_l
    final_img = final_img.astype(np.uint8)

    plt.hist(final_img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.show()
    return final_img


def histogram_smooth(img: np.ndarray, std0: float = 52, m0: float = 128) -> np.ndarray:
    std_ = img.std()
    m_ = img.mean()
    final_img = std0 / std_ * (img - m_) + m0
    plt.hist(final_img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.show()
    return final_img.astype(np.uint8)


def histogram_equalization(img: np.ndarray, split=True, limit_up=255) -> np.ndarray:
    final_img = img.copy()
    if len(img.shape) == 2:
        if split:
            raise Exception
        height, width = img.shape
        sum_ = height * width
        sum_pre = len(np.where(img == 0))
        for i in range(1, 255):
            idx = np.where(img == i)
            sum_pre += len(img[idx])
            value_trans = sum_pre / sum_ * limit_up
            final_img[idx] = value_trans
    elif split:
        height, width, channel = img.shape
        sum_ = height * width
        for i in range(channel):
            img_split = img[..., i]
            final_split = np.zeros_like(img_split)
            sum_pre = len(np.where(img_split == 0))
            for j in range(1, 255):
                idx = np.where(img_split == j)
                sum_pre += len(img_split[idx])
                value_trans = sum_pre / sum_ * limit_up
                final_split[idx] = value_trans
            final_img[..., i] = final_split
    else:
        height, width, channel = img.shape
        sum_ = height * width * channel
        sum_pre = len(np.where(img == 0))
        for i in range(1, 255):
            idx = np.where(img == i)
            sum_pre += len(img[idx])
            value_trans = sum_pre / sum_ * limit_up
            final_img[idx] = value_trans
    plt.hist(final_img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.show()
    return final_img.astype(np.uint8)


def gamma_correction(img: np.ndarray, const: float = 1.0, gamma: float = 2.2) -> np.ndarray:
    final_img = img.copy() / 255.
    final_img = (1 / const * final_img) ** (1 / gamma)
    final_img *= 255
    return final_img.astype(np.uint8)


def nearest_neighbor_interpolation(img: np.ndarray, proportion: float = 1.5) -> np.ndarray:
    height, width = img.shape[:2]
    height_t = int(height * proportion)
    width_t = int(width * proportion)
    y = np.arange(height_t).repeat(width_t).reshape(height_t, -1)
    x = np.tile(np.arange(width_t), (height_t, 1))
    y = np.round(y / proportion).astype(np.int)
    x = np.round(x / proportion).astype(np.int)
    return img[y, x]


def bilinear_interpolation(img: np.ndarray, proportion: float = 1.5) -> np.ndarray:
    height, width = img.shape[:2]
    height_t = int(height * proportion)
    width_t = int(width * proportion)
    y = np.arange(height_t).repeat(width_t).reshape(height_t, -1)
    x = np.tile(np.arange(width_t), (height_t, 1))
    y = y / proportion
    x = x / proportion
    iy = np.floor(y).astype(np.int)
    ix = np.floor(x).astype(np.int)
    iy = np.minimum(iy, height - 1 - 1)
    ix = np.minimum(ix, width - 1 - 1)
    dy = y - iy
    dx = x - ix
    if len(img.shape) == 3:
        dx = np.repeat(np.expand_dims(dx, axis=-1), 3, axis=-1)
        dy = np.repeat(np.expand_dims(dy, axis=-1), 3, axis=-1)
    final_img = (1 - dx) * (1 - dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix + 1] + \
                dy * (1 - dx) * img[iy + 1, ix] + dx * dy * img[iy + 1, ix + 1]
    final_img = np.clip(final_img, 0, 255)
    return final_img.astype(np.uint8)


def bicubic_interpolation(img: np.ndarray, proportion: float = 0.3) -> np.ndarray:
    height, width = img.shape[:2]
    height_t = int(height * proportion)
    width_t = int(width * proportion)
    y = np.arange(height_t).repeat(width_t).reshape(height_t, -1)
    x = np.tile(np.arange(width_t), (height_t, 1))
    y = y / proportion
    x = x / proportion
    iy = np.floor(y).astype(np.int)
    ix = np.floor(x).astype(np.int)

    dy = y - iy
    dx = x - ix
    dys = [dy + 1, dy, 1 - dy, 2 - dy]
    dxs = [dx + 1, dx, 1 - dx, 2 - dx]

    w_sum = np.zeros((height_t, width_t) + img.shape[2:], dtype=np.float32)
    out = np.zeros_like(w_sum, dtype=np.float32)

    def get_weight(x, a=-1.):
        ax = np.abs(x)
        w = np.zeros_like(x)
        idx = np.where(ax <= 1)
        w[idx] = ((a + 2) * np.power(ax, 3) - (a + 3) * np.power(ax, 2) + 1)[idx]
        idx = np.where((1 < ax) & (ax <= 2))
        w[idx] = (a * np.power(ax, 3) - 5 * a * np.power(ax, 2) + 8 * a * ax - 4 * a)[idx]
        return w

    wys = [get_weight(dy) for dy in dys]
    wxs = [get_weight(dx) for dx in dxs]
    if len(img.shape) > 2:
        wys = [np.repeat(np.expand_dims(wy, axis=-1), img.shape[2], axis=-1) for wy in wys]
        wxs = [np.repeat(np.expand_dims(wx, axis=-1), img.shape[2], axis=-1) for wx in wxs]

    for j in range(-1, 3):
        for i in range(-1, 3):
            idy = np.clip(iy + j, 0, height - 1)
            idx = np.clip(ix + i, 0, width - 1)

            wxy = wxs[i + 1] * wys[j + 1]

            w_sum += wxy
            out += wxy * img[idy, idx]

    out /= w_sum
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)


# def afine_transform(img: np.ndarray, move_x=0, move_y=0, rotate_ang=0,
# 						 clock_rotate=0, sharing_dx=0, sharing_dy=0) -> np.ndarray:
# 	height, width = img.shape[:2]
# 	a = 1
# 	b = 0
# 	c = 0
# 	d = 1
# 	tx = 30
# 	ty = -30
# 	trans_mat = np.array([[a, b, tx], [c, d, ty], [0, 1, 1]])
# 	y = np.arange(height).repeat(width).reshape(height, -1)
# 	x = np.tile(np.arange(width), (height, 1))
# 	yx = np.array([x, y, np.ones_like(x)]).T
# 	print(trans_mat.shape, yx.shape)
# 	yt_afine = np.matmul(yx, trans_mat)
# 	y = yt_afine[..., 0]
# 	x = yt_afine[..., 1]
# 	print(y)

# 	print(x)
# 	return yt_afine.astype(np.uint8)


my_function_map = {
    "method21": histogram_normalization,
    "method22": histogram_smooth,
    "method23": histogram_equalization,
    "method24": gamma_correction,
    "method25": nearest_neighbor_interpolation,
    "method26": bilinear_interpolation,
    "method27": bicubic_interpolation,
    "method28": afine_transform,
    # "method29": ,
    # "method30":
}


def test_function():
    img = cv2.imread("../assets/imori.jpg")
    # for func_name in my_function_map:
    img_ = my_function_map['method28'](img)
    cv2.imshow("result", img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_function()
