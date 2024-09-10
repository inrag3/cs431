from PIL import Image, ImageFile
import matplotlib.pyplot as plt

def get_image(path: str) -> ImageFile.ImageFile:
    return Image.open(path)

def get_channels(image: ImageFile.ImageFile) -> tuple[list, list, list]:
    reds: list[int] = []
    greens: list[int] = []
    blues: list[int] = []
    for x in range(image.width):
        for y in range(image.height):
            r, g, b = image.getpixel((x, y))
            reds.append(r)
            greens.append(g)
            blues.append(b)
    return reds, greens, blues

def draw_histogramm(data: list, color: str) -> None:
    values: list[int] = [i for i in range(256)]
    frequency: list[int] = [data.count(i) for i in range(256)]
    plt.bar(values, frequency, color=color, width=1.0)

if __name__ == "__main__":
    image: ImageFile.ImageFile = get_image('chameleon.webp')
    reds, greens, blues = get_channels(image)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    draw_histogramm(reds, 'red')
    plt.subplot(1, 3, 2)
    draw_histogramm(greens, 'green')
    plt.subplot(1, 3, 3)
    draw_histogramm(blues, 'blue')
    plt.show()
