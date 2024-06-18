import tensorflow as tf
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET

# преобразуем папку в tfrecord
fn = "C:/BplaFireMyProg/images"
def load_img(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 256
    img = tf.image.resize(img, (128, 128))
    return img


def create_load_tfrec_for_localizer():
    # формируем список всех xml файлов в папке
    p = [fn + '/' + f for f in listdir(fn) if isfile(join(fn, f)) and f[-1] == 'l']
    # создаем запись
    writer = tf.io.TFRecordWriter('bounding_box_dataset.tfrecord')

    for xml in p:
        tree = ET.parse(xml)  # адрес файла
        root = tree.getroot()  # парсим
        num_objects = len(root) - 6
        cords = []
        w = int(root[4][0].text)  # ширина x
        h = int(root[4][1].text)  # высота y
        for num in range(num_objects):
            object_cords = []
            # нормализуем координаты от -1 до 1, опираясь на исходные координаты
            object_cords.append(int(root[num + 6][4][0].text) / w * 2 - 1)
            object_cords.append(int(root[num + 6][4][1].text) / h * 2 - 1)
            object_cords.append(int(root[num + 6][4][2].text) / w * 2 - 1)
            object_cords.append(int(root[num + 6][4][3].text) / h * 2 - 1)
            cords.append(object_cords)

        img = load_img(root[2].text)
        # готовим данные, представляем в байтовом виде
        serialized_img = tf.io.serialize_tensor(img).numpy()
        serialized_cords = tf.io.serialize_tensor(cords).numpy()
        # собираем экзепмляр
        example = tf.train.Example(features=tf.train.Features(feature={
            'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_img])),
            'cords': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_cords]))
        }))

        # записываем в запись
        writer.write(example.SerializeToString())

    writer.close()

    # и сразу после создания проверка чтения записи
    # прочитаем запись
    dataset = tf.data.TFRecordDataset('bounding_box_dataset.tfrecord')

    def parse_record(record):
        # нужно описать приходящий экземпляр
        # имена элементов как при записи
        feature_description = {
            'img': tf.io.FixedLenFeature([], tf.string),
            'cords': tf.io.FixedLenFeature([], tf.string)
        }
        parsed_record = tf.io.parse_single_example(record, feature_description)
        img = tf.io.parse_tensor(parsed_record['img'], out_type=tf.float32)
        cords = tf.io.parse_tensor(parsed_record['cords'], out_type=tf.float32)
        return img, cords

    # пройдемся по записи и распакуем ее
    dataset = dataset.map(parse_record)

    # проверка тензора
    for i, c in dataset.take(1):
        print(i.shape)
        print(c.shape)

    print("Новая запись локализатора создана")
