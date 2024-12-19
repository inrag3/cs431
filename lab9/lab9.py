import numpy as np
import cv2

# Нормализованное евклидовое расстояние между пикселями
def g(x, y):
   diff = np.sqrt(np.sum((x - y) ** 2)) / np.sqrt(3)
   return max(0, 1 - diff)
    
def growcut(image, state, max_iter=20, window_size=5):
    # state (объект/бекграунд)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image / 255.0
    height, width = image.shape[:2]
    ws = (window_size - 1) // 2

    changes = 1
    n = 0
    state_next = state.copy()
    print(state_next)
    while changes > 0 and n < max_iter:
        changes = 0
        n += 1
        print(n)

        for j in range(width):
            for i in range(height):
                # p - текущий
                C_p = image[i, j] # цвет
                S_p = state[i, j] # сила

                for jj in range(max(0, j - ws), min(j + ws + 1, width)):
                    for ii in range(max(0, i - ws), min(i + ws + 1, height)):
                        # q - который хочет атаковать
                        C_q = image[ii, jj] 
                        S_q = state[ii, jj]
                        print(S_q)
                        gc = g(C_q, C_p)

                        # Вычисление влияния
                        if gc * S_q[1] > S_p[1]:
                            state_next[i, j, 0] = S_q[0]
                            state_next[i, j, 1] = gc * S_q[1]
                            changes += 1
                            break

        state = state_next
    # Возвращается маску объекта
    return state[:, :, 0]

def draw_seeds(event, x, y, flags, param):
    global drawing, current_label, label, strength

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(label, (x, y), 5, current_label, -1)
            cv2.circle(strength, (x, y), 5, 1, -1)
            cv2.circle(display_image, (x, y), 5, (0, 255, 0) if current_label == 1 else (0, 0, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def apply_mask(image, mask):
    output_image = np.zeros_like(image)
    mask = mask * 255
    output_image[mask == 255] = image[mask == 255]

    return output_image

if __name__ == "__main__":
    image = cv2.imread("flower.jpg", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (200, 200))
    label = np.zeros(image.shape)
    strength = np.zeros(image.shape)
    drawing = False
    current_label = 1
    display_image = cv2.imread("flower.jpg")
    display_image = cv2.resize(display_image, (200, 200))

    cv2.namedWindow("Draw Seeds")
    cv2.setMouseCallback("Draw Seeds", draw_seeds)

    while True:
        cv2.imshow("Draw Seeds", display_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('o'):
            current_label = 1
            print("Switched to Object (Green)")

        elif key == ord('b'):
            current_label = -1
            print("Switched to Background (Red)")

        elif key == ord('s'):
            break

    state = np.dstack((label, strength))
    segmented = growcut(image, state)
    image = cv2.imread("flower.jpg")
    image = cv2.resize(image, (200,200))
    result = apply_mask(image, segmented)
    cv2.imwrite("withmask.jpg", result)