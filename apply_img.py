import numpy as np


class ApplyImg:
    def __init__(self):
        pass

    def grayscale(self, image):
        height, width, _ = image.shape

        # Khởi tạo ảnh xám
        grayscale_image = np.zeros((height, width), dtype=np.uint8)

        # Lặp qua từng pixel trong ảnh màu và tính giá trị trung bình
        for i in range(height):
            for j in range(width):
                # Lấy giá trị của các channel màu
                blue, green, red = image[i, j]

                # Tính giá trị trung bình
                gray_value = int(0.299 * red + 0.587 * green + 0.114 * blue)

                # Gán giá trị vào ảnh xám
                grayscale_image[i, j] = gray_value

        return grayscale_image

    def sharpen(self, image):
        # Tạo một kernel sharpening
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])

        # Lấy kích thước của ảnh
        rows, cols, _ = image.shape

        # Tạo một ảnh mới để lưu trữ ảnh đã được tăng cường độ nét
        sharpened_image = np.zeros((rows, cols, 3), dtype=np.uint8)

        # Áp dụng bộ lọc sharpening
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                for k in range(3):  # 3 kênh màu (BGR)
                    sharpened_image[i, j, k] = np.sum(image[i - 1:i + 2, j - 1:j + 2, k] * kernel)

        # Đảm bảo giữ nguyên giá trị pixel trong khoảng [0, 255]
        sharpened_image = np.clip(sharpened_image, 0, 255)

        return sharpened_image

    def erosion(self, image, kernel):
        rows, cols, channels = image.shape
        output_image = np.zeros((rows, cols, channels), dtype=np.uint8)

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                for c in range(channels):
                    output_image[i, j, c] = np.min(image[i - 1:i + 2, j - 1:j + 2, c] * kernel)

        return output_image

    def dilation(self, image, kernel):
        rows, cols, channels = image.shape
        output_image = np.zeros((rows, cols, channels), dtype=np.uint8)

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                for c in range(channels):
                    output_image[i, j, c] = np.max(image[i - 1:i + 2, j - 1:j + 2, c] * kernel)

        return output_image

    def opening(self, image, kernel):
        eroded = self.erosion(image, kernel)
        opened = self.dilation(eroded, kernel)
        return opened

    def closing(self, image, kernel):
        dilated = self.dilation(image, kernel)
        closed = self.erosion(dilated, kernel)
        return closed

    def equalization_histogram(self, image):
        # Lấy kích thước của ảnh
        rows, cols, _ = image.shape

        # Khởi tạo histogram
        histogram = np.zeros(256, dtype=np.int64)

        # Tính histogram
        for i in range(rows):
            for j in range(cols):
                histogram[image[i, j]] += 1

        # Tính histogram cân bằng
        equalization_histogram = np.zeros(256, dtype=np.int64)
        equalization_histogram[0] = histogram[0]
        for i in range(1, 256):
            equalization_histogram[i] = equalization_histogram[i - 1] + histogram[i]

        # Tính histogram cân bằng hóa
        equalization_histogram = np.round(equalization_histogram * 255.0 / equalization_histogram[-1])

        # Tạo ảnh mới
        equalization_image = np.zeros_like(image)

        # Cân bằng histogram
        for i in range(rows):
            for j in range(cols):
                equalization_image[i, j] = equalization_histogram[image[i, j]]

        return equalization_image

    def logarithmic_transformation(self, image):
        # Lấy kích thước của ảnh
        rows, cols,_ = image.shape

        # Khởi tạo ảnh mới
        result = np.zeros_like(image)

        # Tính toán
        for i in range(rows):
            for j in range(cols):
                result[i, j] = 255 * np.log(1 + image[i, j])

        return result

    def gamma_correction(self, image, gamma=1):
        # Lấy kích thước của ảnh
        rows, cols, _ = image.shape

        # Khởi tạo ảnh mới
        result = np.zeros_like(image)

        # Tính toán
        for i in range(rows):
            for j in range(cols):
                result[i, j] = 255 * np.power(image[i, j] / 255.0, gamma)

        return result

    def sobel_filter(self, image):
        # Chuyển ảnh sang ảnh xám nếu chưa phải
        if len(image.shape) == 3:
            gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray_image = image

        # Kernel Sobel cho phát hiện độ nét theo chiều ngang và chiều dọc
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Tính đạo hàm theo chiều ngang và chiều dọc
        gradient_x = np.zeros_like(gray_image)
        gradient_y = np.zeros_like(gray_image)

        for i in range(1, gray_image.shape[0] - 1):
            for j in range(1, gray_image.shape[1] - 1):
                gradient_x[i, j] = np.sum(sobel_x * gray_image[i - 1:i + 2, j - 1:j + 2])
                gradient_y[i, j] = np.sum(sobel_y * gray_image[i - 1:i + 2, j - 1:j + 2])

        # Tính độ nét bằng cách kết hợp kết quả từ cả hai chiều
        gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

        # Chuẩn hóa giá trị pixel về khoảng [0, 255]
        gradient_magnitude = ((gradient_magnitude - np.min(gradient_magnitude)) /
                              (np.max(gradient_magnitude) - np.min(gradient_magnitude)) * 255).astype(np.uint8)

        return gradient_magnitude
