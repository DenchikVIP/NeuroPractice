import tensorflow as tf
#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
def dataprocess():
    tree = ET.parse('C:/BplaFireMyProg/images/fire1.xml')  # адрес файла
    root = tree.getroot()  # парсинг

    # информация в xml файле
    print(root[1].tag, root[1].text)
    print(root[2].tag, root[2].text)
    print(root[4][0].tag, root[4][0].text)
    print(root[4][1].tag, root[4][1].text)

    # сколько обьектов размечено? (6 - кол-во служебных элементов, таких как размер, название и т.д)
    num_objects = len(root) - 6
    print(num_objects)
    for num in range(num_objects):
        print(root[num + 6][0].tag)
        print(root[num + 6][4][0].tag)

    # функция загрузки изображения с помощью tensorflow
    def load_img(img):
        img = tf.io.read_file(img)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32) / 256
        img = tf.image.resize(img, (128, 128))
        return img

    cords = []
    w = int(root[4][0].text)  # ширина x
    h = int(root[4][1].text)  # высота y

    for num in range(num_objects):
        print(root[num + 6][0].text)  # имя обьекта

        object_cords = []
        # нормализуем координаты от -1 до 1, опираясь на исходные координаты
        object_cords.append(int(root[num + 6][4][0].text) / w * 2 - 1)
        object_cords.append(int(root[num + 6][4][1].text) / h * 2 - 1)
        object_cords.append(int(root[num + 6][4][2].text) / w * 2 - 1)
        object_cords.append(int(root[num + 6][4][3].text) / h * 2 - 1)

        cords.append(object_cords)
    print(cords)

    # Проверка правильности вывода изображения на экран
    img = load_img(root[2].text)
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
