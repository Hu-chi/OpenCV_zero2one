import cv2
import numpy as np
from matplotlib import pyplot as plt

def avg_filter(img: np.ndarray, k_size: int=3) -> np.ndarray:
	pad = k_size >> 1
	height, width = img.shape[:2]
	channel = None
	size = (height + pad*2, width + pad*2)
	if len(img.shape) == 3:
		channel = img.shape[2]
		size = (height + pad*2, width + pad*2, channel)
	img_padding = np.zeros(size, dtype=np.float)
	img_padding[pad:pad+height, pad:pad+width] = img.copy().astype(np.float)
	final_img = np.zeros_like(img, dtype=np.float)

	if channel:
		for y in range(height):
			for x in range(width):		
				for c in range(channel):
					final_img[y, x, c] = np.average(img_padding[y:y+k_size, x:x+k_size, c])
	else:
		for y in range(height):
			for x in range(width):
				final_img[y, x] = np.average(img_padding[y:y+k_size, x:x+k_size])

	return final_img.astype(np.uint8)

def motion_filter(img: np.ndarray, k_size: int=3) -> np.ndarray:
	pad = k_size >> 1
	height, width = img.shape[:2]
	channel = None
	size = (height + pad*2, width + pad*2)
	if len(img.shape) == 3:
		channel = img.shape[2]
		size = (height + pad*2, width + pad*2, channel)
	img_padding = np.zeros(size, dtype=np.float)
	img_padding[pad:pad+height, pad:pad+width] = img.copy().astype(np.float)
	final_img = np.zeros_like(img, dtype=np.float)

	motion_filter = np.eye(k_size)
	motion_filter /= motion_filter.sum()

	if channel:
		for y in range(height):
			for x in range(width):		
				for c in range(channel):
					final_img[y, x, c] = np.sum(motion_filter * img_padding[y:y+k_size, x:x+k_size, c])
	else:
		for y in range(height):
			for x in range(width):
				final_img[y, x] = np.sum(motion_filter * img_padding[y:y+k_size, x:x+k_size])

	return final_img.astype(np.uint8)

def max_min_filter(img: np.ndarray, k_size: int=3) -> np.ndarray:
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	pad = k_size >> 1
	height, width = img.shape[:2]
	size = (height + pad*2, width + pad*2)
	img_padding = np.zeros(size, dtype=np.float)
	img_padding[pad:pad+height, pad:pad+width] = img.copy().astype(np.float)
	final_img = np.zeros_like(img, dtype=np.float)

	for y in range(height):
		for x in range(width):
			final_img[y, x] = np.max(img_padding[y:y+k_size, x:x+k_size]) \
							- np.min(img_padding[y:y+k_size, x:x+k_size])

	return final_img.astype(np.uint8)

def custom_filter(img: np.ndarray, k_size: int=3, filter=None) -> np.ndarray:
	if filter is None:
		raise Exception
	for dim in filter.shape:
		if dim != k_size:
			raise Exception
	if len(img.shape) == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	pad = k_size >> 1
	height, width = img.shape[:2]
	size = (height + pad*2, width + pad*2)
	img_padding = np.zeros(size, dtype=np.float)
	img_padding[pad:pad+height, pad:pad+width] = img.copy().astype(np.float)
	final_img = np.zeros_like(img, dtype=np.float)
	
	for y in range(height):
		for x in range(width):
			final_img[y, x] = np.sum(filter * img_padding[y:y+k_size, x:x+k_size])
	final_img[final_img < 0] = 0
	final_img[final_img > 255] = 255

	return final_img.astype(np.uint8)


def differential_filter(img: np.ndarray, direction="horizontal") -> np.ndarray:
	diff_filter_ = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
	if direction == "horizontal":
		diff_filter_ = diff_filter_.T

	return custom_filter(img, 3, diff_filter_)
		
def sobel_filter(img: np.ndarray, direction="horizontal") -> np.ndarray:
	sobel_filter_ = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
	if direction == "horizontal":
		sobel_filter_ = sobel_filter_.T

	return custom_filter(img, 3, sobel_filter_)

def prewitt_filter(img: np.ndarray, direction="horizontal") -> np.ndarray:
	prewitt_filter_ = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
	if direction == "horizontal":
		prewitt_filter_ = prewitt_filter_.T

	return custom_filter(img, 3, prewitt_filter_)

def laplacian_filter(img: np.ndarray) -> np.ndarray:
	laplacian_filter_ = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

	return custom_filter(img, 3, laplacian_filter_)

def emboss_filter(img: np.ndarray) -> np.ndarray:
	emboss_filter_ = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

	return custom_filter(img, 3, emboss_filter_)

def LoG_filter(img: np.ndarray, k_size: int=5, sigma: float=3) -> np.ndarray:
	LoG_filter_ = np.zeros((k_size, k_size), dtype=np.float)
	pad = k_size >> 1

	for i in range(-pad, k_size-pad):
		for j in range(-pad, k_size-pad):
			sq_ij = i**2 + j**2
			LoG_filter_[i+pad, j+pad] = (sq_ij - sigma**2) * np.exp(-sq_ij / (2 *(sigma**2)))
	LoG_filter_ /= LoG_filter_.sum()
	print(LoG_filter_)
	return custom_filter(img, 5, LoG_filter_)

def plot_histogram(img: np.ndarray) -> np.ndarray:
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
	plt.show()
	return img


my_function_map = {
	"method11": avg_filter,
	"method12": motion_filter,
	"method13": max_min_filter,
	"method14": differential_filter,
	"method15": sobel_filter,
	"method16": prewitt_filter,
	"method17": laplacian_filter,
	"method18": emboss_filter,
	"method19": LoG_filter,
	"method20": plot_histogram
}

def test_function():
	img = cv2.imread("../assets/imori.jpg")
	# for func_name in my_function_map:
	img_ = my_function_map['method20'](img)
	cv2.imshow("result", img_)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	test_function()