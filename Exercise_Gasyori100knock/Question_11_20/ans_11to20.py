import cv2
import numpy as np

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


def differential_filter(img: np.ndarray, k_size: int=3, direction="horizontal") -> np.ndarray:
	diff_filter = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
	if direction == "horizontal":
		diff_filter = diff_filter.T

	return custom_filter(img, k_size, diff_filter)
		
def sobel_filter(img: np.ndarray, k_size: int=3, direction="horizontal") -> np.ndarray:
	diff_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
	if direction == "horizontal":
		diff_filter = diff_filter.T

	return custom_filter(img, k_size, diff_filter)

def prewitt_filter(img: np.ndarray, k_size: int=3, direction="horizontal") -> np.ndarray:
	diff_filter = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
	if direction == "horizontal":
		diff_filter = diff_filter.T

	return custom_filter(img, k_size, diff_filter)


my_function_map = {
	"method11": avg_filter,
	"method12": motion_filter,
	"method13": max_min_filter,
	"method14": differential_filter,
	"method15": sobel_filter,
	"method16": prewitt_filter
}

def test_function():
	img = cv2.imread("../assets/imori.jpg")
	# for func_name in my_function_map:
	img_ = my_function_map['method16'](img)
	cv2.imshow("result", img_)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	test_function()