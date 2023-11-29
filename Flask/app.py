from flask import Flask, render_template, request, send_file
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

import cv2

def process_image(input_image, algorithm):
    img = cv2.imread(input_image)

    if algorithm == 'grayscale':
        # Chuyển ảnh thành ảnh xám
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif algorithm == 'histogram_equalization':
        # Cân bằng histogram
        processed_img = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    elif algorithm == 'negative':
        # Ảnh âm bảng
        processed_img = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif algorithm == 'thresholding':
        # Ngưỡng
        _, processed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 128, 255, cv2.THRESH_BINARY)
    elif algorithm == 'logarithmic_transformation':
        # Logarithmic transformation
        processed_img = (np.log1p(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) * 255 / np.log1p(256)).astype(np.uint8)
    elif algorithm == 'power_law_transform':
        # Power law transform
        gamma = 0.5  # Thay đổi giá trị gamma theo ý muốn
        processed_img = (255 * (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255) ** gamma).astype(np.uint8)
    elif algorithm == 'bit_plane_slicing':
        # Bit plane slicing
        bit = 7  # Thay đổi giá trị bit theo ý muốn
        processed_img = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) & (1 << bit)).astype(np.uint8) * 255
    elif algorithm == 'spatial_filtering':
        # Bộ lọc không gian (ví dụ: Gaussian Blur)
        processed_img = cv2.GaussianBlur(img, (5, 5), 0)
    elif algorithm == 'edges_processing':
        # Xử lý cạnh (ví dụ: Canny edge detection)
        processed_img = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)
    elif algorithm == 'laplacian_filtered_image':
        # Laplacian filtered image
        processed_img = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
        processed_img = cv2.convertScaleAbs(processed_img)
    elif algorithm == 'sharpened_image':
        # Sharpened image (ví dụ: sharpening kernel)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        processed_img = cv2.filter2D(img, -1, kernel)
    elif algorithm == 'sobel_filter':
        # Sobel filter
        processed_img = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 1, ksize=3)
        processed_img = cv2.convertScaleAbs(processed_img)
    elif algorithm == 'sobel_filter_with_thresholding':
        # Sobel filter with thresholding
        processed_img = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 1, ksize=3)
        processed_img = cv2.convertScaleAbs(processed_img)
        _, processed_img = cv2.threshold(processed_img, 50, 255, cv2.THRESH_BINARY)
    elif algorithm == 'points_detection':
        # Points detection (ví dụ: Shi-Tomasi corner detection)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img, (x, y), 3, 255, -1)
        processed_img = img
    elif algorithm == 'lines_detection':
        # Lines detection (ví dụ: HoughLines)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), 255, 2)
        processed_img = img
    elif algorithm == 'erosion':
        # Erosion
        kernel = np.ones((5, 5), np.uint8)
        processed_img = cv2.erode(img, kernel, iterations=1)
    elif algorithm == 'dilation':
        # Dilation
        kernel = np.ones((5, 5), np.uint8)
        processed_img = cv2.dilate(img, kernel, iterations=1)
    elif algorithm == 'opening':
        # Opening
        kernel = np.ones((5, 5), np.uint8)
        processed_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif algorithm == 'closing':
        # Closing
        kernel = np.ones((5, 5), np.uint8)
        processed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    elif algorithm == 'boundary_extraction':
        # Boundary extraction
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        processed_img = cv2.filter2D(img, -1, kernel)
    elif algorithm == 'region_filling':
        # Region filling (ví dụ: flood fill)
        processed_img = img.copy()
        h, w = img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(processed_img, mask, (0, 0), 255)

    return processed_img


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Nhận file ảnh từ form
        file = request.files['image']
        
        # Nhận thuật toán từ form
        algorithm = request.form['algorithm']

        # Thực hiện xử lý ảnh
        processed_image = process_image(file, algorithm)

        # Lưu ảnh sau xử lý vào bộ nhớ đệm
        img_byte_array = io.BytesIO()
        processed_image.save(img_byte_array, format='PNG')
        img_byte_array = img_byte_array.getvalue()

        # Trả về ảnh đã xử lý cho người dùng
        return send_file(io.BytesIO(img_byte_array),
                         mimetype='image/png',
                         as_attachment=True,
                         download_name='processed_image.png')

    # Nếu là GET request hoặc request không thành công, hiển thị trang web
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
