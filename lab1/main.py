from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_image(path: str):
    return np.array(Image.open(path))

def get_video(path: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(path)

def apply(image, func):
    value = func(
        image[:, :, 2].astype(float),
        image[:, :, 1].astype(float),
        image[:, :, 0].astype(float)
    )
    gray_image = np.stack((value, value, value), axis=-1).astype(np.uint8)
    return gray_image

def color_difference(first, second):
    return cv2.absdiff(first, second)

def difference(first, second):
    # Вычисляем разность между двумя изображениями
    diffs = color_difference(first, second)
    return diffs

def to_grayscale(image):
    return apply(image, lambda r, g, b: 0.3 * r + 0.59 * g + 0.11 * b)

if __name__ == "__main__":
    image = get_image('mona_lisa.jpg')
    image_clone = image.copy()
    image = apply(image, lambda r, g, b: (r + g + b) // 3)
    image_clone = apply(image_clone, lambda r, g, b: 0.3 * r + 0.59 * g + 0.11 * b)
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(image_clone)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    image_difference = difference(image, image_clone)
    plt.imshow(image_difference)
    plt.axis('off')
    plt.show()
    # Второе задание
    video = get_video("butterfly.mp4")
    _, frame = video.read()

    previous = to_grayscale(frame)

    while True:

        ret, frame = video.read()

        if not ret:
            break

        next = to_grayscale(frame)
        image = difference(previous, next)
        previous = next
        cv2.imshow('Video', image)
        cv2.waitKey(50)
    video.release()
    cv2.destroyAllWindows()