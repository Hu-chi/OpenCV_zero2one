import cv2
import numpy as np
from matplotlib import pyplot as plt

def histogram_normalization(img: np.ndarray, limit_l: int=0, limit_r: int=255) -> np.ndarray:
	if (limit_r < limit_l) :
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

def histogram_smooth(img: np.ndarray, std0: float=52, m0: float=128) -> np.ndarray:
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

def gamma_correction(img: np.ndarray, const: float=1.0, gamma: float=2.2) -> np.ndarray:
	final_img = img.copy() / 255.
	final_img = (1 / const * final_img) ** (1 / gamma)
	final_img *= 255
	return final_img.astype(np.uint8)

def nearest_neighbor_interpolation(img: np.ndarray, proportion: float) -> np.ndarray:
	# TODO(TEST, huchi)
	if proportion < 1:
		throw Exception
	height, width = img.shape[:2]
	height_t = int(height * proportion)
	width_t = int(width * proportion)
	y = np.arange(height_t).repeat(width_t).reshape(width_t, -1) # idx_y
	x = np.tile(np.arange(width_t), (height_t, 1))
	y = np.round(y / proportion).astype(np.int)
	x = np.round(x / proportion).astype(np.int)
	return img[y, x]

my_function_map = {
	"method21": histogram_normalization,
	"method22": histogram_smooth,
	"method23": histogram_equalization,
	"method24": gamma_correction,
	"method25"
}



def test_function():
	img = cv2.imread("../assets/imori_dark.jpg")
	# for func_name in my_function_map:
	img_ = my_function_map['method24'](img)
	cv2.imshow("result", img_)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	test_function()