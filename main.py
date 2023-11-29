from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.params import Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import tempfile

from apply_img import ApplyImg

app = FastAPI()

# Configuring Jinja2 templates
templates = Jinja2Templates(directory="templates")

def apply_filter(img, processing_mode):
    if processing_mode == "grayscale":
        # Code xử lý ảnh giữ nguyên
        return ApplyImg().grayscale(img)
    elif processing_mode == "sharpen":
        # Code xử lý ảnh bằng Sharpening
        return ApplyImg().sharpen(img)
    elif processing_mode == "opening":
        # Code xử lý ảnh bằng Opening
        return ApplyImg().opening(img, np.ones((3,3,3),np.uint8))
    elif processing_mode == "closing":
        # Code xử lý ảnh bằng Closing
        return ApplyImg().closing(img, np.ones((3,3,3),np.uint8))
    # Region Filling

    elif processing_mode == "boundary-extraction":
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Áp dụng GaussianBlur để làm mờ ảnh và giảm nhiễu
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Sử dụng Canny để phát hiện biên
        edges = cv2.Canny(blurred_image, 50, 150)
        return edges
    # Soble Filter with Thresholding
    elif processing_mode == "erosion":
        # Code xử lý ảnh bằng Erosion
        kernel = np.ones((3,3,3),np.uint8)
        img = ApplyImg().erosion(img, kernel)
        return img
    elif processing_mode == "dilation":
        # Code xử lý ảnh bằng Dilation
        kernel = np.ones((3,3,3),np.uint8)
        img = ApplyImg().dilation(img, kernel)
        return img
    elif processing_mode == "sobel":
        # Code xử lý ảnh bằng Sobel
        img = ApplyImg().sobel_filter(img)
        return img
    elif processing_mode == "equalization-histogram":
        # Code xử lý ảnh bằng Equalization Histogram
        img = ApplyImg().equalization_histogram(img)
        return img
    elif processing_mode == "logarithmic-transformation":
        img = ApplyImg().logarithmic_transformation(img)
        return img
    elif processing_mode == "gamma-correction":
        img = ApplyImg().gamma_correction(img)
        return img
    else:
        raise HTTPException(status_code=400, detail="Invalid processing mode")

# def apply_img_using_cv2(img, processing_mode):
#     if processing_mode == "grayscale":
#         # Code xử lý ảnh giữ nguyên
#         return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     elif processing_mode == "sharpen":
#         # Code xử lý ảnh bằng Sharpening
#         kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#         return cv2.filter2D(img, -1, kernel)
#     elif processing_mode == "opening":
#         # Code xử lý ảnh bằng Opening
#         kernel = np.ones((3,3),np.uint8)
#         return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#     elif processing_mode == "closing":
#         # Code xử lý ảnh bằng Closing
#         kernel = np.ones((3,3),np.uint8)
#         return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#     # Region Filling
#
#     elif processing_mode == "boundary-extraction":
#         gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         # Áp dụng GaussianBlur để làm mờ ảnh và giảm nhiễu
#         blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
#
#         # Sử dụng Canny để phát hiện biên
#         edges = cv2.Canny(blurred_image, 50, 150)
#         return edges
#     # Soble Filter with Thresholding
#     elif processing_mode == "erosion":
#         # Code xử lý ảnh bằng Erosion
#         kernel = np.ones((3,3),np.uint8)
#         return cv2.erode(img, kernel, iterations=1)
#     elif processing_mode == "dilation":
#         # Code xử lý ảnh bằng Dilation
#         kernel = np.ones((3,3),np.uint8)
#         return cv2.dilate(img, kernel, iterations=1)
#     elif processing_mode == "sobel":
#         # Code xử lý ảnh bằng Sobel
#         gray_image = cv2.cvtColor
#         return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
#     elif processing_mode == "equalization-histogram":
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         # Áp dụng equalizeHist cho kênh sáng (luma)
#         equalized = cv2.equalizeHist(gray)
#         # Tạo ảnh màu mới với kênh sáng được cập nhật
#         equalized_img = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
#
#         return equalized_img
#     elif processing_mode == "logarithmic-transformation":
#         # Code xử lý ảnh bằng Logarithmic Transformation
#         return np.uint8(np.log1p(img)*255/np.log1p(255))
#     elif processing_mode == "gamma-correction":
#         # Code xử lý ảnh bằng Gamma Correction
#         return np.uint8(np.power(img/float(np.max(img)), 2)*255)
#     else:
#         raise HTTPException(status_code=400, detail="Invalid processing mode")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process-image/")
async def process_image(request: Request, file: UploadFile = File(...), processing_mode: str = Query('')):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    processed_img = apply_img_using_cv2(img, processing_mode)

    _, processed_img_encoded = cv2.imencode('.png', processed_img)
    processed_img_bytes = processed_img_encoded.tobytes()

    # Tạo file tạm thời và lưu ảnh xử lý vào đó
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(processed_img_bytes)
        processed_image_path = temp_file.name

    return FileResponse(
        processed_image_path,
        media_type="image/png",
    )
