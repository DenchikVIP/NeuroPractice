import tensorflow as tf
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Combobox
import tkinter as tk
from PIL import Image, ImageTk
import io
from dataprocessing import dataprocess

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from os import listdir
from os.path import isfile, join

global path_for_one
localizator = tf.keras.models.load_model('my_bb_model.keras')
classifier = tf.keras.models.load_model('my_classifier.keras')

window = Tk()
window.geometry("1240x720")
window.resizable(False, False)
window.title("Классификатор пожара")
window.iconbitmap(default="bplaicon.ico")
window.configure(bg='silver')
image_field_raw = Canvas(bg="darkgray", height=500, width=500)
image_field_raw.place(relx=0.05, rely=0.2, relheight=0.6, relwidth=0.4)
title1 = Label(image_field_raw, font=30, text=f"Загруженное изображение")
title1.pack(anchor=N)
image_field_ready = Canvas(bg="darkgray", height=500, width=500)
image_field_ready.place(relx=0.55, rely=0.2, relheight=0.6, relwidth=0.4)
title2 = Label(image_field_ready, font=30, text=f"Результат распознавания")
title2.pack(anchor=N)
frame = Frame(window, bg='darkgray')
frame.place(relheight=0.1, relwidth=1)
frame1 = Frame(window, bg='darkgray')
frame1.place(relheight=0.1, relwidth=1, rely=0.9)

class ConsoleRedirector(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

# функция работы с нейросетями: подготовка изображения, детекция
# на выходе три массива: координаты (10,4) , классы (10) , вероятности (10)
def detect_objects(image):
    # подготовка картинки
    image = tf.cast(image, dtype=tf.float32) / 256
    small_image = tf.image.resize(image, (128, 128))
    big_image = tf.image.resize(image, (1024, 1024))
    image_exp = tf.expand_dims(small_image, axis=0)

    # локализация
    bb_cords = localizator(image_exp)
    bb_cords = tf.squeeze(bb_cords, axis=0)

    # нормализация по размеру картинки
    bb_cords = (bb_cords + 1) / 2 * 128
    bb_cords = tf.reshape(bb_cords, [10, 3])

    # разделяем на элементы
    fxmin, ymin, fxmax = tf.split(bb_cords, 3, axis=1)

    # нормализуем
    xmin = tf.minimum(fxmin, fxmax)
    xmax = tf.maximum(fxmin, fxmax)

    xmin = tf.clip_by_value(xmin, 0, 128)
    ymin = tf.clip_by_value(ymin, 0, 128)

    size = xmax - xmin

    # сумма координаты и размера должны быть <= 128
    xsize = tf.clip_by_value(size, 1, 128 - xmin)
    ysize = tf.clip_by_value(size, 1, 128 - ymin)

    # нарезаем и собираем в массив (10, 32, 32, 3)
    ymin *= 8
    xmin *= 8
    ysize *= 8
    xsize *= 8
    for n in range(10):
        ii = tf.image.crop_to_bounding_box(big_image, int(ymin[n][0]), int(xmin[n][0]), int(ysize[n][0]),
                                           int(xsize[n][0]))
        ii = tf.image.resize(ii, (128, 128))
        ii = tf.expand_dims(ii, axis=0)
        if n == 0:
            cropped = ii
        else:
            cropped = tf.concat([cropped, ii], axis=0)

    # классифицируем
    probs = classifier(cropped)

    probs = probs.numpy()

    # считаем метки класса (индекс наибольшего среди вероятностей)
    ma = np.amax(probs, axis=1)
    ma = np.expand_dims(ma, axis=1)
    _, classes = np.where(probs == ma)

    # берем ту вероятность, которая наибольшая
    res_probs = []
    for a in range(10):
        res_probs.append(probs[a][classes[a]])

    # собираем координаты в нормальный вид
    cords = tf.concat([xmin / 8, ymin / 8, xmin / 8 + xsize / 8, ymin / 8 + ysize / 8], axis=1)
    cords = cords.numpy()

    return cords, classes, res_probs

# th - вероятность, ниже которой рамки не отображаются
namespace = {0: 'NOTHING', 1: 'fire'}

def visualize(in_image, cords, classes, probs, th=0.5):
    big_image = tf.image.resize(in_image, (1024, 1024)).numpy() / 256

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2

    for i in range(len(cords)):
        if classes[i] != 0 and probs[i] >= th:
            # введем цвета для всех обьектов
            if classes[i] == 1:
                color = (1, 0, 0)
            if classes[i] == 2:
                color = (0, 1, 0)
            text = namespace[classes[i]] + ' ' + str(probs[i] * 100) + '%'

            org = (int(cords[i][0]) * 8, int(cords[i][1]) * 8 - 10)
            # рисуем текст и квадраты
            big_image = cv2.putText(big_image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
            big_image = cv2.rectangle(big_image, (int(cords[i][0]) * 8, int(cords[i][1]) * 8),
                                      (int(cords[i][2]) * 8, int(cords[i][3]) * 8), color, 5)

    return big_image

# функция обьединяет две рамки одного класса в одну, если они пересекаются
# tau - порог IoU этих рамок чтобы их обьединить (0.1 - все подряд, 0.9 - только очень близкие)
# функцию можно применять несколько раз, результат улучшится

def prettify(cords, classes, probs, tau=0.2):
    newcords = []
    newclasses = []
    newprobs = []

    for i1 in range(len(classes)):

        if classes[i1] != 0:
            found = False
            for i2 in range(len(classes)):
                if classes[i2] != 0 and i1 != i2:

                    # вычислим IoU, самый надежный способ определить совпадение
                    x_overlap = max(0, min(cords[i1][2], cords[i2][2]) - max(cords[i1][0], cords[i2][0]))
                    y_overlap = max(0, min(cords[i1][3], cords[i2][3]) - max(cords[i1][1], cords[i2][1]))
                    inter = x_overlap * y_overlap
                    area1 = (cords[i1][2] - cords[i1][0]) * (cords[i1][3] - cords[i1][1])
                    area2 = (cords[i2][2] - cords[i2][0]) * (cords[i2][3] - cords[i2][1])
                    union = area1 + area2 - inter
                    IoU = inter / union
                    if IoU > tau:
                        # считаем среднее по всем координатам между двух рамок
                        newcord = [(cords[i1][0] + cords[i2][0]) // 2, (cords[i1][1] + cords[i2][1]) // 2,
                                   (cords[i1][2] + cords[i2][2]) // 2, (cords[i1][3] + cords[i2][3]) // 2]
                        newcords.append(newcord)

                        newclasses.append(classes[i1])
                        newprobs.append(probs[i1])

                        # обнуляем класс, чтобы больше не крутить эту рамку
                        classes[i1] = 0
                        classes[i2] = 0
                        found = True

            # если ни с чем не обьединили, так и оставляем
            if found == False:
                newcords.append(cords[i1])
                newclasses.append(classes[i1])
                newprobs.append(probs[i1])

    return newcords, newclasses, newprobs

def detect_fire_in_image(path_for_one):
    image = cv2.imread(path_for_one)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cords, classes, probs = detect_objects(image)
    for i in range(1):
        cords, classes, probs = prettify(cords, classes, probs, 0.8)
    result = visualize(image, cords, classes, probs, 0.5)

    buf = io.BytesIO()
    plt.figure()
    plt.imshow(result)
    plt.axis('off')  # Скрываем оси
    plt.savefig(buf, format='png')
    buf.seek(0)

    original_image = Image.open(buf)
    resized_image = original_image.resize(
        (int(original_image.width * 1.35), int(original_image.height * 1.15)))
    image_tk = ImageTk.PhotoImage(resized_image)

    image_field_ready.create_image(-192, -65, anchor=NW, image=image_tk, tags='Старое')
    image_field_ready.image_tk = image_tk  # Сохраняем ссылку на изображение

    # Закрываем буфер
    buf.close()

def loadimage():
    global image
    global path_for_one
    path_for_one = filedialog.askopenfilename(filetypes=[("Изображения", "*.png;*.jpg;*.jpeg")])
    if path_for_one:
        pil_image = Image.open(path_for_one)
        pil_image = pil_image.resize((500, 500))
        image = ImageTk.PhotoImage(pil_image)
        image_field_raw.create_image(0, 0, anchor=NW, image=image, tags='Старое')
def detect():
    global path_for_one
    detect_fire_in_image(path_for_one)


def train_window_open():
    window1 = Tk()
    window1.geometry("720x510")
    window1.resizable(False, False)
    window1.title("Окно тренировки модели нейросети")
    window1.iconbitmap(default="bplaicon.ico")
    window1.configure(bg='silver')

    close_button = Button(window1, text="Закрыть окно", command=lambda: window1.destroy())
    close_button.pack(anchor=NE, expand=0)

    processdataforonefile = Button(window1, font=40, text=f"Информация о картинке", command=dataprocess)
    processdataforonefile.pack(anchor=NW)

    text = tk.Text(window1)
    text.pack(anchor='center')
    sys.stdout = ConsoleRedirector(text)



def load_model_data():
    da = 0


load_image_but = Button(frame, font=40, text=f"Загрузить изображение", command=loadimage)
load_image_but.grid(column=0, row=0, padx=10, pady=20)

load_model_but = Button(frame, font=40, text=f"Загрузить модель для обучения", command=load_model_data)
load_model_but.grid(column=1, row=0, padx=10)

detect_but = Button(frame, font=40, text=f"Распознать", bg='Green', fg='black', command=detect)
detect_but.grid(column=2, row=0, padx=10)

train_window_but = Button(frame1, font=40, text=f"Открыть окно для обучения", command=train_window_open)
train_window_but.grid(column=0, row=0, padx=10, pady=20)

window.mainloop()
