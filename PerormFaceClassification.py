# develop a classifier for the 5 Celebrity Faces Dataset
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
from matplotlib import image

# загрузка лиц
data = load('faces_dataset.npz')
testX_faces = data['arr_2']

# загрузка эмбедингов
data = load('faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# нормалицая входных векторов
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# лэйблы кодируют таргеты
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# подгоняем модель
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

# selection = choice([i for i in range(testX.shape[0])])
print('Укажите номер файла')
selection = int(input())
selected_face_pixels = testX_faces[selection]
selected_face_emb = testX[selection]
selected_face_class = testy[selection]
selected_face_name = out_encoder.inverse_transform([selected_face_class])

# предикшн для выбранного лица
samples = expand_dims(selected_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

# получить имя прогнозируемого целого числа класса и вероятность этого прогноза
class_index = yhat_class[0]
class_probability = yhat_prob[0, class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)

# вывод
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % selected_face_name[0])
pyplot.imshow(selected_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()
