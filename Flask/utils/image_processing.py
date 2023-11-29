from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np
import math
import cv2

def grayscale_conversion(image):
    return image.convert('L')

def histogram_equalization(image):
    # Cân bằng histogram
    image_array = np.array(image)
    hist, bins = np.histogram(image_array.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    equalized_image = cdf[image_array]
    return Image.fromarray(equalized_image)

def negative_image(image):
    return ImageOps.invert(image)

def thresholding(image, threshold):
    return image.point(lambda p: 255 if p > threshold else 0)

def logarithmic_transformation(image):
    c = 255 / math.log(256)
    return image.point(lambda p: int(c * math.log(p + 1)))

def power_law_transform(image, gamma):
    return image.point(lambda p: int(255 * (p / 255) ** gamma))

def bit_plane_slicing(image, bit):
    image_array = np.array(image)
    return Image.fromarray((image_array & (1 << bit)).astype(np.uint8) * 255)

def spatial_filtering(image):
    # Bộ lọc không gian (ví dụ: Gaussian Blur)
    return image.filter(ImageFilter.GaussianBlur(radius=2))

def edges_processing(image):
    # Xử lý cạnh (ví dụ: ImageFilter.FIND_EDGES)
    return image.filter(ImageFilter.FIND_EDGES)

def laplacian_filtered_image(image):
    # Laplacian filtered image
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return image.filter(ImageFilter.Kernel((3, 3), kernel.flatten(), scale=1, offset=128))

def sharpened_image(image):
    # Sharpened image (ví dụ: ImageFilter.SHARPEN)
    return image.filter(ImageFilter.SHARPEN)

def sobel_filter(image):
    # Sobel filter
    gradient_x = image.filter(ImageFilter.FIND_EDGES)
    gradient_y = image.rotate(90).filter(ImageFilter.FIND_EDGES).rotate(-90)
    return ImageOps.grayscale(ImageOps.colorize(gradient_x, gradient_y, "black"))

def sobel_filter_with_thresholding(image, threshold):
    # Sobel filter with thresholding
    sobel_image = sobel_filter(image)
    return sobel_image.point(lambda p: 255 if p > threshold else 0)

def points_detection(image):
    # Points detection (ví dụ: Harris Corner Detection)
    gray = image.convert('L')
    gray_np = np.array(gray)
    corners = cv2.cornerHarris(gray_np, 2, 3, 0.04)
    corners = Image.fromarray((corners > 0.01 * corners.max()) * 255).convert('L')
    return corners

def lines_detection(image):
    # Lines detection (ví dụ: HoughLines)
    gray = image.convert('L')
    edges = gray.filter(ImageFilter.FIND_EDGES)
    lines = edges.convert('L').transform((image.width, image.height), Image.HOUGH)
    return lines

def erosion(image):
    # Erosion
    return image.filter(ImageFilter.MinFilter(size=3))

def dilation(image):
    # Dilation
    return image.filter(ImageFilter.MaxFilter(size=3))

def opening(image):
    # Opening (Erosion followed by Dilation)
    return dilation(erosion(image))

def closing(image):
    return erosion(dilation(image))

def boundary_extraction(image):
    edges = image.filter(ImageFilter.FIND_EDGES)
    return ImageOps.invert(edges)

def region_filling(image):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    mask = Image.new('L', (width + 2, height + 2), 0)
    draw.floodfill((0, 0), 255, mask=mask)
    return image

# Example Usage:
input_image_path = "image.jpg"
image = Image.open(input_image_path)

processed_image = grayscale_conversion(image)
processed_image.show()
