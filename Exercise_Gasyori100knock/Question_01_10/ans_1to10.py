import cv2
import numpy as np

def channel_swap_BGR2RGB(img: np.ndarray) -> np.ndarray:
	if len(img.shape) < 3:
		raise Exception
	r, b = img[:, :, 2].copy(), img[:, :, 0].copy()
	img[:, :, 0], img[:, :, 2] = r, b
	return img

def img_trans_BGR2Gray(img: np.ndarray) -> np.ndarray:
	if len(img.shape) < 3:
		raise Exception
	img = img[:, :, 0] * 0.0722 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.2126
	return img.astype(np.uint8)

def bi_thresholding(img: np.ndarray, threshold: int=128) -> np.ndarray:
	if len(img.shape) == 3:
		img = img_trans_BGR2Gray(img)
	img[img >= threshold] = 255
	img[img < threshold] = 0
	return img

def otsu_thresholding(img: np.ndarray) -> np.ndarray:
	if len(img.shape) == 3:
		img = img_trans_BGR2Gray(img)
	best_threshold = 0
	max_bvar = 0
	height, width = img.shape[:2]
	for threshold_ in range(1, 256):
		v0 = img[img < threshold_]
		m0 = np.mean(v0) if len(v0) > 0 else 0.
		w0 = len(v0) / (height * width)
		v1 = img[img > threshold_]
		m1 = np.mean(v1) if len(v1) > 0 else 0.
		w1 = len(v1) / (height * width)
		bvar = w0 * w1 * ((m0 - m1) ** 2)
		if bvar > max_bvar:
			max_bvar = bvar
			best_threshold = threshold_
	return bi_thresholding(img, best_threshold)

def img_trans_BGR2HSV(img: np.ndarray) -> np.ndarray:
	"""
		seems incorrect 
	"""
	img = img.astype(np.float32) / 255
	max_bgr = np.max(img, axis=-1)
	min_bgr = np.min(img, axis=-1)
	min_idx = np.argmin(img, axis=-1)
	hue = np.zeros_like(max_bgr)
	hue[max_bgr == min_bgr] = 0
	for c1, c2, c3, bias in zip([0, 2, 1], [1, 0, 2], [2, 1, 0], [60, 180, 300]):
		idx = np.where(min_idx == c1)
		hue[idx] = 60 * (img[..., c2][idx] - img[..., c3][idx]) / (max_bgr[idx] - min_bgr[idx]) + bias
	saturation = max_bgr - min_bgr
	value = max_bgr
	final_img = np.zeros_like(img)
	for i, j in zip([0, 1, 2], [hue, saturation, value]):
		final_img[..., i] = j
	return final_img

def img_trans_HSV2BGR(img: np.ndarray) -> np.ndarray:
	"""
		failed function 
	"""
	hue_ = img[..., 0] / 60
	saturation_ = img[..., 1]
	value = img[..., 2]
	tmp_x = saturation_ * (1 - np.abs(hue_ % 2 - 1))
	tmp_z = np.zeros_like(tmp_x)
	tmp_c = saturation_
	choice = [	
				[tmp_z, tmp_x, tmp_c], 
			  	[tmp_z, tmp_c, tmp_x], 
			  	[tmp_x, tmp_c, tmp_z], 
			  	[tmp_c, tmp_x, tmp_z], 
			 	[tmp_c, tmp_z, tmp_x], 
			  	[tmp_x, tmp_z, tmp_c]
			]
	final_img = np.zeros_like(img)
	for i in range(6):
		idx = np.where((i <= hue_) & (hue_ < i + 1))
		for c in range(3):
			final_img[..., c][idx] = (value - tmp_c)[idx] + choice[i][c][idx]
	
	final_img = (final_img * 255).astype(np.uint8)
	return final_img

def color_compress(img: np.ndarray) -> np.ndarray:
	img = img // 64 * 64 + 32
	return img

def pooling(img: np.ndarray, method="average", filter_size=(4, 4)) -> np.ndarray:
	pool_func = np.mean
	method = method.lower()
	if "average" == method:
		pool_func = np.mean
	elif "max" == method:
		pool_func = np.max
	elif "min" == method:
		pool_func = np.min
	else:
		raise Exception("pool method %s hasn't been finished" % method)
	height, width = img.shape[:2]
	channel = None
	if len(img.shape) == 3:
		channel = img.shape[2]
	new_h = height // filter_size[0]
	new_w = width // filter_size[1]
	for y in range(new_h):
		for x in range(new_w):
			h_slice = slice(filter_size[0]*y, filter_size[0]*(y+1))
			w_slice = slice(filter_size[1]*x, filter_size[1]*(x+1))
			if not channel:
				img[h_slice, w_slice] = pool_func(img[h_slice, w_slice]).astype(np.int)
			else:
				for c in range(channel):
					img[h_slice, w_slice, c] = pool_func(img[h_slice, w_slice, c]).astype(np.int)
	return img

def gaussian_filter(img: np.ndarray, k_size: int=3, sigma: int=1.3) -> np.ndarray:
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
	gaussian_kernel = np.zeros((k_size, k_size), dtype=np.float)
	for x in range(-pad, k_size-pad):
		for y in range(-pad, k_size-pad):
			gaussian_kernel[y+pad, x+pad] = np.exp(-(x**2 + y**2) / (2 * (sigma**2)))
	gaussian_kernel /= gaussian_kernel.sum()

	if channel:
		for y in range(height):
			for x in range(width):
				for c in range(channel):
					final_img[y, x, c] = np.sum(gaussian_kernel * img_padding[y:y+k_size, x:x+k_size, c])
	else:
		for y in range(height):
			for x in range(width):
				final_img[y, x] = np.sum(gaussian_kernel * img[y:y+k_size, x:x+k_size])
				
	return final_img.astype(np.uint8)



def median_filter(img: np.ndarray, k_size: int=3) -> np.ndarray:
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
					final_img[y, x, c] = np.median(img_padding[y:y+k_size, x:x+k_size, c])
	else:
		for y in range(height):
			for x in range(width):
				final_img[y, x] = np.median(img_padding[y:y+k_size, x:x+k_size])

	return final_img.astype(np.uint8)

my_function_map = {
	"method1": channel_swap_BGR2RGB,
	"method2": img_trans_BGR2Gray,
	"method3": bi_thresholding,
	"method4": otsu_thresholding,
	"method5": lambda x: cv2.cvtColor(img_trans_BGR2HSV(x), cv2.COLOR_HSV2BGR),
	# this one goes wrong:
	# "method5": lambda x: img_trans_HSV2BGR(cv2.cvtColor(x, cv2.COLOR_BGR2HSV)),
	"method6": color_compress,
	"method7": pooling,
	"method8": lambda x: pooling(x, "max"),
	"method9": gaussian_filter,
	"method10": median_filter
}


def test_function():
	img = cv2.imread("../assets/imori.jpg")
	for func_name in my_function_map:
		img_ = my_function_map[func_name](img)
		cv2.imshow("result", img_)
		cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	test_function()






"""
method 5 origin code 
import cv2
import numpy as np

# Read image
img = cv2.imread("imori.jpg").astype(np.float32) / 255.

# RGB > HSV
out = np.zeros_like(img)

max_v = np.max(img, axis=2).copy()
min_v = np.min(img, axis=2).copy()
min_arg = np.argmin(img, axis=2)

H = np.zeros_like(max_v)

H[np.where(max_v == min_v)] = 0
## if min == B
ind = np.where(min_arg == 0)
H[ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
## if min == R
ind = np.where(min_arg == 2)
H[ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
## if min == G
ind = np.where(min_arg == 1)
H[ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300
    
V = max_v.copy()
S = max_v.copy() - min_v.copy()

# Transpose Hue
H = (H + 180) % 360

# HSV > RGB

C = S
H_ = H / 60
X = C * (1 - np.abs( H_ % 2 - 1))
Z = np.zeros_like(H)

vals = [[Z,X,C], [Z,C,X], [X,C,Z], [C,X,Z], [C,Z,X], [X,Z,C]]

for i in range(6):
    ind = np.where((i <= H_) & (H_ < (i+1)))
    out[..., 0][ind] = (V-C)[ind] + vals[i][0][ind]
    out[..., 1][ind] = (V-C)[ind] + vals[i][1][ind]
    out[..., 2][ind] = (V-C)[ind] + vals[i][2][ind]

out[np.where(max_v == min_v)] = 0
out = (out * 255).astype(np.uint8) 

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""