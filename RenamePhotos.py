import os

x = 0

for i in os.listdir("D:\\pythonProject\\faces_dataset\\val\\edu"):
    if i.endswith(".jpg"):
        os.rename(i, "photo" + str(x) + ".jpg")
        x += 1
