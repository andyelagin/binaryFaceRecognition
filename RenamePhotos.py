import os

index = 20
path = "D:/pythonProject/faces_dataset/val/no_edu"
for file in os.listdir(path):
    if file.endswith(".jpg"):
        os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index), '.jpg'])))
        index += 1
