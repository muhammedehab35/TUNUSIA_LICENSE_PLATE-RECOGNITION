import os
import numpy as np
from skimage import measure
from skimage.io import imread
from skimage.filters import threshold_otsu
from PIL import Image
import scipy.fftpack 
import pandas as pd
import cv2
import matplotlib.pyplot as plt


IMAGE_DIR = "tayma_data/IMAGES"
LABEL_DIR = "tayma_data/labels"  



def image_extraction(image_dir, label_dir, channel):
    raw_data = []
    labels = []
    image_files = os.listdir(image_dir)
    
    for image_file in image_files:
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + ".txt"  
            label_path = os.path.join(label_dir, label_file)
            img = cv2.imread(img_path, channel)
            raw_data.append(img)
            with open(label_path, 'r') as file:
                label = file.read().strip()  
                labels.append(label)
    return raw_data, labels


def supprimerPetitsObjets(imgBW, areaPixels):
    imgBWcopy = imgBW.copy()
    _, contours, _ = cv2.findContours(imgBWcopy.copy(),
                                      cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_SIMPLE)
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, (0, 0, 0), -1)
    
    return imgBWcopy



def homomorphic_filter(csv_files):
    filtered_data = []
    labels = []
    i = 0
    for _, row in csv_files.iterrows():
        i = i + 1
        try:
            file = row['image_path']
            label = row['lp']
            filename = "data/"+ file.split(sep='/')[1] + '/' + file.split(sep='/')[2]
            print(i, filename)
            img = cv2.imread(filename, 0)
            rows = img.shape[0]
            cols = img.shape[1]
            img = img[:, 59:cols-20]
            imgLog = np.log1p(np.array(img, dtype="float") / 255)
            M = 2*rows + 1
            N = 2*cols + 1
            sigma = 10
            (X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M))
            centerX = np.ceil(N/2)
            centerY = np.ceil(M/2)
            gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2
            Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma))
            Hhigh = 1 - Hlow
            HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
            HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())
            If = scipy.fftpack.fft2(imgLog.copy(), (M, N))
            Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M, N)))
            Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M, N)))
            gamma1 = 0.3
            gamma2 = 1.5
            Iout = gamma1*Ioutlow[0:rows, 0:cols] + gamma2*Iouthigh[0:rows, 0:cols]
            Ihmf = np.expm1(Iout)
            Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
            Ihmf2 = np.array(255*Ihmf, dtype="uint8")
            Ithresh = Ihmf2 < 65
            Ithresh = 255*Ithresh.astype("uint8")
            Iclear = imclearborder(Ithresh, 5)
            Iopen = bwareaopen(Iclear, 120)
            filtered_data.append(Iopen)
            labels.append(label)
        except:
            pass
    return filtered_data, labels




def extrctionMSER():
    img = cv2.imread('data/crop_h1/I00000.png')
    mser = cv2.MSER_create()
    img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()
    regions = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    cv2.polylines(vis, hulls, 1, (0, 255, 0))
    cv2.namedWindow('img', 0)
    cv2.imshow('img', vis)
    while cv2.waitKey() != ord('q'):
        continue
    cv2.destroyAllWindows()
    cv2.imshow('Homomorphic filtered output', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cca v1
    image = cv2.imread('data/crop_h1/I00000.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if numPixels > 300:
            mask = cv2.add(mask, labelMask)
    cv2.imshow('Filtered output', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_segments(data):
    segments = list()
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] == 255:
                segments.append(get_component(data, i, j))
    return segments


def print_segments(segments):
    individual = []
    for segment in segments:
        top_left_row = 100000000
        top_left_col = 100000000
        bottom_right_row = -1
        bottom_right_col = -1
        for (x, y) in segment:
            top_left_col = min(top_left_col, y)
            top_left_row = min(top_left_row, x)
            bottom_right_col = max(bottom_right_col, y)
            bottom_right_row = max(bottom_right_row, x)
        img = Image.new('L', (bottom_right_row - top_left_row + 1, bottom_right_col - top_left_col + 1))
        pixel = img.load()
        for i in range(bottom_right_row - top_left_row + 1):
            for j in range(bottom_right_col - top_left_col + 1):
                pixel[i, j] = 0
        for i in segment:
            pixel[i[0] - top_left_row, i[1] - top_left_col] = 255
        individual.append(img)
    return individual


def convert_image_to_numpy(individual):
    characters = []
    for i in individual:
        inter_mediate = np.array(i)
        characters.append(inter_mediate)
    for i in characters:
        cv2.imshow('CHAR', i)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return characters


def save_filtered_data(copy_filtered_data, labels):
    for i in range(0, len(copy_filtered_data)):
        cv2.imwrite("images/filtered/" + labels[i] + "-" + str(i) + ".png", copy_filtered_data[i])
        cv2.imshow(str(labels[i]) + " " + str(i), copy_filtered_data[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def filtered_image_extraction(files):
    clean_data = []
    labels = []
    for file in files:
        img = cv2.imread("images/filtered/" + file, 1)
        label = file.split(sep="-")[0]
        clean_data.append(img)
        labels.append(label)
    return clean_data, labels

def noise_removal(copy_X, index):
    factor = 0
    for i in index:
        del copy_X[i - factor]
        factor = factor + 1
    for i in range(0, len(copy_X)):
        file = "images/individual/" + str(i) + ".png"
        cv2.imwrite(file, copy_X[i])
        

def flip_and_rotate(): 
    clean = []
    individual_files = os.listdir('images/individual')
    for i in range(0, len(individual_files)):
        img = cv2.imread('images/individual/' + individual_files[i])
        img = cv2.flip(img, 1)
        clean.append(img)
    for i in range(0, len(clean)):
        cv2.imwrite('images/clean/' + str(i) + '.png', clean[i])
    return clean


def final_extraction(folder_list):
    X = []
    Y = []
    for folder in folder_list:
        file_list = os.listdir('images/segregated/' + folder)
        for file in file_list:
            img = cv2.imread('images/segregated/' + folder + '/' + file, 0)
            X.append(img)
            Y.append(folder)
    return X, Y


def determine_max_row_and_column_size(data):
    max_row_size = 0
    max_col_size = 0
    for i in data:
        size = np.shape(i)
        if size[0] > max_row_size:
            max_row_size = size[0]
        if size[1] > max_col_size:
            max_col_size = size[1]
    return max_row_size, max_col_size


def image_padding_by_resize(data, pad_x, pad_y):
    out = []
    for i in data:
        u = cv2.resize(i, (pad_x, pad_y))
        out.append(u)
    return out



def show_sample():
    license_plate = imread("data/crop_h1/I00000.png", as_grey=True)/255.0
    print(license_plate.shape)
    gray_car_image = license_plate * 255
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(gray_car_image, cmap="gray")
    threshold_value = threshold_otsu(gray_car_image)
    binary_car_image = gray_car_image > threshold_value
    ax2.imshow(binary_car_image, cmap="gray")
    print(binary_car_image)



def show_homomorphed_sample(image, n_index):
    cv2.imshow('Homomorphic filtered output', image[n_index])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def preparing_data():
    global CSV_FILES
    show_sample()
    filtered_data, filtered_labels = homomorphic_filter(CSV_FILES)
    show_homomorphed_sample(filtered_data, 578)
    segments_list = []
    for each_plate in filtered_data:
        corner_y = np.shape(each_plate)[1] - 1
        corner_x = np.shape(each_plate)[0] - 1
        get_component(each_plate, 0, 0)
        get_component(each_plate, 0, corner_y)
        get_component(each_plate, corner_x, 0)
        get_component(each_plate, corner_x, corner_y)
        segments_list.append(get_segments(each_plate))
    individual_list = []
    for segments in segments_list:
        individual_list.append(print_segments(segments))
    individual_images = []
    for plate in individual_list:
        for char in plate:
            individual_images.append(np.array(char))
    labels = []
    for i in filtered_labels:
        for j in i:
            labels.append(j)
    copy_individual_images = individual_images
    index = []
    for i in range(0, len(copy_individual_images)):
        if(np.shape(copy_individual_images[i])[0] > 40
           or np.shape(copy_individual_images[i])[0] < 15):
            index.append(i)
    index = []
    for i in range(0, len(copy_individual_images)):
        if(np.shape(copy_individual_images[i])[1] < 15
           or np.shape(copy_individual_images[i])[1] > 100):
            index.append(i)
    for i in index:
        cv2.imshow(str(i), copy_individual_images[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
