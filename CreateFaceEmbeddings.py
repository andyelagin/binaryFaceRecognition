# вычичляет эмбединги лиц
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model


# получает эмбединг для одного лица
def get_embedding(model, face_pixels):
    # масштабирует значеия пикселей
    face_pixels = face_pixels.astype('float32')
    # Стандартизирует значений пикселей канала
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    # преобразование лица в один образец
    samples = expand_dims(face_pixels, axis=0)

    yhat = model.predict(samples)
    return yhat[0]


# заграужает файл распознаных лиц
data = load('faces_dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

# загружает предтренированную модель facenet
model = load_model('facenet_keras.h5')
print('Loaded Model')

# конвертирует каждое лицо в ОБУЧАЮЩИЙ сет для эмбединга
newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)

# конвертирует каждое лицо в ТЕСТОВЫЙ сет для эмбединга
newTestX = list()
for face_pixels in testX:
    embedding = get_embedding(model, face_pixels)
    newTestX.append(embedding)
newTestX = asarray(newTestX)
print(newTestX.shape)

savez_compressed('faces-embeddings.npz', newTrainX, trainy, newTestX, testy)
