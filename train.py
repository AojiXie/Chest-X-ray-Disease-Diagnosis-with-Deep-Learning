# USAGE
# python train.py --dataset dataset --model smallvggnet.model --labelbin mlb.pickle


# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import pandas as pd

'''
import shutil
source = '/home/shiyong/Shiyong/GATech/CSE6250/Project/data/Images/images (12)'
dest1 = '/home/shiyong/Shiyong/GATech/CSE6250/Project/data/Images/images'
files = os.listdir(source)
for f in files:
    shutil.move(os.path.join(source,f), dest1)

'''

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset (i.e., directory of images)")
ap.add_argument("-c", "--labelcsv", required=True,help="path to label csv file")
ap.add_argument("-f", "--trainvalidationlist", required=True,help="path to train validation list file")
ap.add_argument("-m", "--model", required=True,help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (128, 128, 1)

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())


load = True
if load:
    with open("/Users/xieaoji/Desktop/cse6250/CSE6250-Project-master/20k_image_data_balanced.pckl", "rb") as fp:
        data = pickle.load(fp)
    with open("/Users/xieaoji/Desktop/cse6250/CSE6250-Project-master/20k_image_labels_balanced.pckl", "rb") as fp:
        labels = pickle.load(fp)
else:
    train_validation_frames = open(args["trainvalidationlist"]).read().splitlines()

    # grab the image paths and randomly shuffle them
    print("[INFO] loading images...")
    img_dir = args["dataset"]
    imagePaths = [os.path.join(img_dir, f) for f in train_validation_frames if os.path.isfile(os.path.join(img_dir, f))]
    random.seed(100)
    random.shuffle(imagePaths)


    def split_label(label):
        label_list = label.split('|')
        return label_list


    # initialize the data and labels
    data = []
    labels = []

    df = pd.read_csv(args["labelcsv"], delimiter=',')
    image_labels = [(x[0], split_label(x[1])) for x in df.values]  # if x[0] in set(train_validation_frames)]
    img_to_labels_dict = dict((img_name, labels) for img_name, labels in image_labels)

    no_finding_img_to_labels_dict = {}
    with_finding_img_to_labels_dict = {}
    train_valid_set = set(train_validation_frames)
    for img, label in img_to_labels_dict.items():
        if img not in train_valid_set:
            continue
        if 'No Finding' in label:
            no_finding_img_to_labels_dict[img] = label
        else:
            with_finding_img_to_labels_dict[img] = label

    # loop over the input images
    total_cnt = len(with_finding_img_to_labels_dict)
    print("loading with finding images ...")
    for cnt, image_name in enumerate(with_finding_img_to_labels_dict.keys()):
        imagePath = os.path.join(img_dir, image_name)
        if cnt % 100 == 0:
            print("processed {}/{}".format(cnt, total_cnt))
        if cnt >= 18000:
            break
        image_name = os.path.basename(imagePath)
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)
        data.append(image)
        labels.append(img_to_labels_dict[image_name])

    total_cnt = len(no_finding_img_to_labels_dict)
    print("loading no finding images ...")
    for cnt, image_name in enumerate(no_finding_img_to_labels_dict.keys()):
        imagePath = os.path.join(img_dir, image_name)
        if cnt % 100 == 0:
            print("processed {}/{}".format(cnt, total_cnt))
        if cnt >= 2000:
            break
        image_name = os.path.basename(imagePath)
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)
        data.append(image)
        labels.append(img_to_labels_dict[image_name])

    with open("/Users/xieaoji/Desktop/cse6250/CSE6250-Project-master/20k_image_data_balanced.pckl", "wb") as fp:
        pickle.dump(data, fp)
    with open("/Users/xieaoji/Desktop/cse6250/CSE6250-Project-master/20k_image_labels_balanced.pckl", "wb") as fp:
        pickle.dump(labels, fp)
    exit(0)


# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
#print("[INFO] data matrix: {} images ({:.2f}MB)".format(len(imagePaths), data.nbytes / (1024 * 1000.0)))

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i + 1, label))

print("[INFO] train_test_split...")
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.2, random_state=100)

print("[INFO] image data generator...")
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
    width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
    depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
    finalAct="sigmoid")
print(model.summary())

# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(mlb))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])

#'''