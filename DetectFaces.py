# # confirm mtcnn was installed correctly
# import mtcnn
# # print version
# print(mtcnn.__version__)

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
    # обнаружение лица на изображении
    results = detector.detect_faces(pixels)
    # рамка для первого распознаного лица
    x1, y1, width, height = results[0]['box']
    # Sometimes the library will return a negative pixel index. fix this by taking
    # the absolute value of the coordinates.
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # извлечение лица
    face = pixels[y1:y2, x1:x2]
    # изменение размера пикселей под размер модели
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        face = extract_face(path)
        # store
        faces.append(face)
    return faces


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)


# load train dataset
trainX, trainy = load_dataset('faces_dataset/train/')
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset('faces_dataset/val/')
# save arrays to one file in compressed format
savez_compressed('faces_dataset.npz', trainX, trainy, testX, testy)
