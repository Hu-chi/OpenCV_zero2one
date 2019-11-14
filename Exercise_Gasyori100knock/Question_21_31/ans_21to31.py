import cv2
import numpy as np
from matplotlib import pyplot as plt


def histogram_normalization(img: np.ndarray, limit_l: int = 0, limit_r: int = 255) -> np.ndarray:
    if limit_r < limit_l:
        limit_l, limit_r = limit_r, limit_l
    height, width = img.shape[:2]
    channel = None if len(img.shape) < 3 else img.shape[2]
    img_min = img.min()
    img_max = img.max()
    if img_min == img_max:
        return np.full_like(img, limit_l + limit_r >> 1)

    final_img = img.copy()
    final_img = np.clip(final_img, limit_l, limit_r)
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

    def get_img(img_pre: np.ndarray, img_post: np.ndarray) -> np.ndarray:
        sum_pre = len(np.where(img_pre == 0))
        for i in range(1, 255):
            idx = np.where(img_pre == i)
            sum_pre += len(img_pre[idx])
            value_trans = sum_pre / sum_ * limit_up
            img_post[idx] = value_trans
        return img_post

    if len(img.shape) == 2:
        if split:
            raise Exception
        height, width = img.shape
        sum_ = height * width
        final_img = get_img(img, final_img)
    elif split:
        height, width, channel = img.shape
        sum_ = height * width
        for i in range(channel):
            img_split = img[..., i]
            final_split = np.zeros_like(img_split)
            final_split = get_img(img_split, final_split)
            final_img[..., i] = final_split
    else:
        height, width, channel = img.shape
        sum_ = height * width * channel
        final_img = get_img(img, final_img)
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
    y = np.arange(height_t).repeat(width_t).reshape(height_t, -1) / proportion
    x = np.tile(np.arange(width_t), (height_t, 1)) / proportion
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
    y = np.arange(height_t).repeat(width_t).reshape(height_t, -1) / proportion
    x = np.tile(np.arange(width_t), (height_t, 1)) / proportion
    iy = np.floor(y).astype(np.int)
    ix = np.floor(x).astype(np.int)

    dy = y - iy
    dx = x - ix
    dys = [dy + 1, dy, 1 - dy, 2 - dy]
    dxs = [dx + 1, dx, 1 - dx, 2 - dx]

    w_sum = np.zeros((height_t, width_t) + img.shape[2:], dtype=np.float32)
    out = np.zeros_like(w_sum, dtype=np.float32)

    def get_weight(d_, a=-1.):
        ax = np.abs(d_)
        w = np.zeros_like(d_)
        idx_ = np.where(ax <= 1)
        w[idx_] = ((a + 2) * np.power(ax, 3) - (a + 3) * np.power(ax, 2) + 1)[idx_]
        idx_ = np.where((1 < ax) & (ax <= 2))
        w[idx_] = (a * np.power(ax, 3) - 5 * a * np.power(ax, 2) + 8 * a * ax - 4 * a)[idx_]
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


def affine_transform(img: np.ndarray, move_x: int = 0, move_y: int = 0, rotate_ang: float = 0, scale_x: float = 1,
                     scale_y: float = 1, clock_rotate: bool = False, sharing_dx: int = 0, sharing_dy: int = 0
                     ) -> np.ndarray:
    if clock_rotate:
        rotate_ang = -rotate_ang
    height, width = img.shape[:2]
    height = int(scale_y * height)
    width = int(scale_x * width)
    sin_theta = np.sin(rotate_ang)
    cos_theta = np.cos(rotate_ang)

    # TODO(huchi): Actually I don't know what trans_mat is when affine and rotating both occurs
    trans_mat = np.array([[cos_theta, -sin_theta + sharing_dx / height, move_x],
                          [sin_theta + sharing_dy / width, cos_theta, move_y]
                          ]).astype(np.float)
    y = np.arange(height).repeat(width).reshape(height, -1)
    x = np.tile(np.arange(width), (height, 1))

    y_px_p1 = np.array([x.T, y.T, np.ones_like(x).T]).T
    y_px_p1 = np.expand_dims(y_px_p1, axis=-1).astype(np.float)

    yx1_affine = np.matmul(trans_mat, y_px_p1).squeeze()
    y_trans = np.clip(yx1_affine[..., 1], 0, height - 1).astype(np.int)
    x_trans = np.clip(yx1_affine[..., 0], 0, width - 1).astype(np.int)
    final_img = np.zeros((height, width) + img.shape[2:], dtype=np.float)
    y = (y / scale_y).astype(np.int)
    x = (x / scale_x).astype(np.int)

    final_img[y_trans, x_trans] = img[y, x]
    return final_img.astype(np.uint8)


my_function_map = {
    "method21": histogram_normalization,
    "method22": histogram_smooth,
    "method23": histogram_equalization,
    "method24": gamma_correction,
    "method25": nearest_neighbor_interpolation,
    "method26": bilinear_interpolation,
    "method27": bicubic_interpolation,
    "method28": lambda x: affine_transform(x, move_x=30, move_y=-30),
    "method29": lambda x: affine_transform(x, scale_x=1.3, scale_y=0.8),
    "method30": lambda x: affine_transform(x, rotate_ang=-np.pi / 6, move_x=-20, move_y=40),
    # TODO(hucih): affine function seems not
    "method31": lambda x: affine_transform(x, sharing_dx=30, sharing_dy=30)
}


def test_function():
    img = cv2.imread("../assets/imori.jpg")
    # for func_name in my_function_map:
    img_ = my_function_map['method31'](img)
    cv2.imshow("result", img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_function()
