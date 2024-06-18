import tensorflow as tf
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import random

# преобразуем папку в tfrecord
fn = "C:/BplaFireMyProg/images"


# формируем список всех xml файлов в папке


def check_xml_list():
    p = [fn + '/' + f for f in listdir(fn) if isfile(join(fn, f)) and f[-1] == 'l']
    print(p[:20])


# функция загрузки изображения
def load_img(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 256
    img = tf.image.resize(img, (128, 128))
    return img


def create_load_tfrec_for_classifier():


    # преобразуем папку в tfrecord для классификарора

    namespace = {'NOTHING': 0, 'fire': 1}

    p = [fn + '/' + f for f in listdir(fn) if isfile(join(fn, f)) and f[-1] == 'l']
    # создаем запись
    writer = tf.io.TFRecordWriter('classifier_dataset.tfrecord')

    def saveinrecord(img, name):
        # готовим данные, представляем в байтовом виде
        serialized_img = tf.io.serialize_tensor(img).numpy()
        serialized_name = tf.io.serialize_tensor(name).numpy()
        # собираем экзепмляр
        example = tf.train.Example(features=tf.train.Features(feature={
            'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_img])),
            'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_name]))
        }))

        # записываем в запись
        writer.write(example.SerializeToString())

    for xml in p:
        tree = ET.parse(xml)  # адрес файла
        root = tree.getroot()  # парсим
        num_objects = len(root) - 6

        w = int(root[4][0].text)  # ширина x
        h = int(root[4][1].text)  # высота y

        img = load_img(root[2].text)
        for num in range(num_objects):
            xmin = tf.clip_by_value(int(int(root[num + 6][4][0].text) / w * 1024), 0, 1024)
            ymin = tf.clip_by_value(int(int(root[num + 6][4][1].text) / h * 1024), 0, 1024)
            xmax = tf.clip_by_value(int(int(root[num + 6][4][2].text) / w * 1024), 0, 1024)
            ymax = tf.clip_by_value(int(int(root[num + 6][4][3].text) / h * 1024), 0, 1024)

            offset_height = ymin
            offset_width = xmin

            target_height = ymax - ymin
            target_width = xmax - xmin

            cropped = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
            cropped = tf.image.resize(cropped, (128, 128))

            name = namespace[root[num + 6][0].text]

            saveinrecord(cropped, name)

        # создадим и рамки фона
        counter = 0
        goal = 5

        while counter < goal:
            # сгенерим случайные координаты рамки
            gxmin = random.randint(1, 900)
            gymin = random.randint(1, 900)
            gxsize = random.randint(10, 100)
            gysize = random.randint(10, 100)

            gxmax = gxmin + gxsize
            gymax = gymin + gysize

            # для случаев, когда рамка пересекается с реальной
            notintersect = True

            for num in range(num_objects):
                xmin = tf.clip_by_value(int(int(root[num + 6][4][0].text) / w * 1024), 0, 1024)
                ymin = tf.clip_by_value(int(int(root[num + 6][4][1].text) / h * 1024), 0, 1024)
                xmax = tf.clip_by_value(int(int(root[num + 6][4][2].text) / w * 1024), 0, 1024)
                ymax = tf.clip_by_value(int(int(root[num + 6][4][3].text) / h * 1024), 0, 1024)

                x_overlap = tf.maximum(0, tf.minimum(gxmax, xmax) - tf.maximum(gxmin, xmin))
                y_overlap = tf.maximum(0, tf.minimum(gymax, ymax) - tf.maximum(gymin, ymin))
                if x_overlap > 0 and y_overlap > 0:
                    notintersect = False
                    break

            if notintersect:
                cropped = tf.image.crop_to_bounding_box(img, gymin, gxmin, gysize, gxsize)
                cropped = tf.image.resize(cropped, (128, 128))
                name = 0
                saveinrecord(cropped, name)
                counter += 1

    writer.close()

    # прочитаем запись
    dataset = tf.data.TFRecordDataset('classifier_dataset.tfrecord')

    def parse_record(record):
        # нужно описать приходящий экземпляр
        # имена элементов как при записи
        feature_description = {
            'img': tf.io.FixedLenFeature([], tf.string),
            'name': tf.io.FixedLenFeature([], tf.string)
        }
        parsed_record = tf.io.parse_single_example(record, feature_description)
        img = tf.io.parse_tensor(parsed_record['img'], out_type=tf.float32)
        name = tf.io.parse_tensor(parsed_record['name'], out_type=tf.int32)
        return img, name

    # пройдемся по записи и распакуем ее
    dataset = dataset.map(parse_record)

    # Проверка тензора
    for i, c in dataset.take(1):
        print(i.shape)
        print(c)

    print("Новая запись классификатора создана")
