import numpy as np
import cv2


def apply_image_transformations(image, transformation):
    """
    Apply image transformations:
    - GRAY: convert to grayscale
    - RGB: convert to RGB numpy array
    - GRAY_SOBEL: grayscale + Sobel X/Y as 2 channels
    - RGB_SOBEL: RGB + Sobel X/Y for each channel, stacked as 6 channels
    """
    # Ensure image is RGB
    np_image = np.array(image.convert("RGB"))

    match transformation:
        case "GRAY":
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
            return gray  # single channel

        case "RGB":
            return cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)  # 3 channels

        case "SOBEL":
            scale = 1.5
            delta = 0
            ddepth = cv2.CV_16S

            src = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
            src = cv2.GaussianBlur(src, (3, 3), 0)
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

            grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
            grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
            #
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)

            grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

            return grad

        case "CANNY":
            img = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
            img_blur = cv2.GaussianBlur(img, (5, 5), 0)

            v = np.median(img_blur)
            std = np.std(img_blur)
            sigma = np.clip(std / 128.0, 0.05, 0.33)

            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edges = cv2.Canny(img_blur, lower, upper)

            return edges

        case _:
            return cv2.cvtColor(np_image, cv2.IMREAD_UNCHANGED)

def get_image_channels_from_filter(filter_name) -> int | None:
    match filter_name:
        case "GRAY":
            return 1
        case "RGB":
            return 3
        case "SOBEL":
            return 1
        case "CANNY":
            return 1
        case _:
            return None