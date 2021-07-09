# обнаружение лица
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN


# извлекает единственное лицо из фотографии
def extract_face(filename, required_size=(160, 160)):
    # загрузка изображения из файла
    image = Image.open(filename)
    # конвертация в RGB, если надо
    image = image.convert('RGB')
    # конвертация в массив
    pixels = asarray(image)

    # создание детектора
    detector = MTCNN()
    # обнаружение лиц на изображении
    results = detector.detect_faces(pixels)

    # рамка для первого распознаного лица
    x1, y1, width, height = results[0]['box']
    # баг фикс
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    # извлечение лица
    face = pixels[y1:y2, x1:x2]

    # изменение размера пикселей под размер модели
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)

    return face_array


# загружает все лица в список для данной директории
def load_faces(directory):
    faces = list()
    for filename in listdir(directory):
        # путь
        path = directory + filename
        # получает лицо
        face = extract_face(path)
        faces.append(face)
    return faces


# берет имя каталога и обнаруживает лица для каждого подкаталога, то биж лица
# и приваивает лейблы каждому обнаруженому лици
def load_dataset(directory):
    X, y = list(), list()
    for subdir in listdir(directory):
        path = directory + subdir + '/'
        # пропускает любые файлы, которые могут быть в каталоге
        if not isdir(path):
            continue
        # загружает все лица в поддерикторию
        faces = load_faces(path)
        # создание лейблов
        labels = [subdir for _ in range(len(faces))]

        print('>loaded %d examples for class: %s' % (len(faces), subdir))

        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)


# загружает обучающие данные
trainX, trainy = load_dataset('faces_dataset/train/')
print(trainX.shape, trainy.shape)

# загружает тестовые данные
testX, testy = load_dataset('faces_dataset/val/')

# сохраняет маасивы в файл
savez_compressed('faces_dataset.npz', trainX, trainy, testX, testy)
