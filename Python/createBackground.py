import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from os import listdir
from os.path import isfile, join
import shutil
import os

class Annotation:
    def __init__(self, line):
        tmp = line.split(";")
        self.image_name = tmp[0]
        self.top_left = (int(tmp[1]), int(tmp[2]))
        self.bottom_right = (int(tmp[3]), int(tmp[4]))
        self.sign_class = tmp[5]




def IOU(img1, img2):
    xA = max(img1[0], img2[0])
    yA = max(img1[1], img2[1])
    xB = min(img1[2], img2[2])
    yB = min(img1[3], img2[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (img1[2] - img1[0] + 1) * (img1[3] - img1[1] + 1)
    boxBArea = (img2[2] - img2[0] + 1) * (img2[3] - img2[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def getAnnotationsForImage(image_name, ann_list):
    result = []
    for ann in ann_list:
        if ann.image_name == image_name:
            result.append(ann)
    return result

def createBackgroundsFromDataset(path, result_path):
    first_x_images = 10
    image_size = 48
    roi_rate = 0.2
    #step_size = int(image_size/2)
    step_size = image_size
    ann_path = path + "gt.txt"
    annotation_list = []
    new_file_counter = 0
    f = open(ann_path, "r")
    for line in f:
        annotation = Annotation(line)
        annotation_list.append(annotation)
    f.close()

    for i in range(first_x_images):
        zeros_in_file_name = 5 - len(str(i))
        file_name = ""
        for j in range(zeros_in_file_name):
            file_name += "0"
        file_name += str(i) + ".ppm"

        image_ann_list = getAnnotationsForImage(file_name, annotation_list)
        img = cv2.imread(path + file_name)
        background_list = createBacgroundsFromImage(img, image_ann_list, image_size, step_size, roi_rate, roi_rate)
        for j in range(len(background_list)):
            next_file_name = "00000_"
            zeros_in_next_file_name = 5 - len(str(new_file_counter))
            for k in range(zeros_in_next_file_name):
                next_file_name += "0"
            next_file_name += str(new_file_counter) + ".ppm"
            cv2.imwrite(result_path+next_file_name, background_list[j])
            new_file_counter += 1 

def createBacgroundsFromImage(img, list_of_ground_trouth, img_size, step_size, top_roi_rate, bottom_roi_rate):
    h, w, c = img.shape
    result_list = []
    total_count = 0
    ok_count = 0
    contains_gt_count = 0
    out_of_roi_count = 0
    diff = int(img_size / step_size)
    range_i = int(w/step_size) - diff
    range_j = int(h/step_size) - diff
    for i in range(range_i):
        for j in range(range_j):
            top = i * step_size
            left = j * step_size
            coords = (top, left, top + img_size, left + img_size)
            top_left = (coords[0], coords[1])
            bottom_right = (coords[2], coords[3])
            #print("top-left: {0}".format(top_left))
            #print("bottom-right: {0}".format(bottom_right))
            sample = img[coords[1]:coords[3], coords[0]:coords[2]]
            total_count += 1
            if sample.shape != (img_size, img_size, 3):
                tmp = cv2.rectangle(img.copy(), top_left, bottom_right, (0, 255, 0), 2)
                print("i:" + str(i))
                print("j:" + str(j))
                print(coords)
                cv2.imshow("tmp", tmp)
                cv2.waitKey()
                cv2.destroyAllWindows()

            if isImageContainsGroundTrouth(coords, list_of_ground_trouth):
                contains_gt_count += 1
                continue
            if isImageOutOfRoi(coords, h, top_roi_rate, bottom_roi_rate):
                out_of_roi_count += 1
                continue
            result_list.append(sample)
            ok_count += 1
    print("Ezen a képen összesen {0} ennyi képrészlet volt, ebből {1} jó, {2} roi-n kívüli, {3} sok táblát tartalmazó kép található"
          .format(total_count, ok_count, out_of_roi_count, contains_gt_count))
    return result_list


def createFolders(new_set_location):
    for folder_counter in range(43):
        folder_num = createFileName("", folder_counter, extension="/")
        new_folder_name = new_set_location + folder_num
        os.mkdir(new_folder_name)


def createAugmentedImages(min_samples_num, orig_set, new_set_location):
    for folder_counter in range(43):
        folder_num = createFileName("", folder_counter, extension="")
        actual_folder = orig_set + str(folder_num) + "/"
        destination_folder = new_set_location + str(folder_num) + "/"
        files_in_folder = [f for f in listdir(actual_folder) if isfile(join(actual_folder, f))]

        counter = 0
        # for f in files_in_folder:
        #     old_file = actual_folder + f
        #     new_file_name = createFileName(destination_folder, counter)
        #     shutil.copy(old_file, new_file_name)
        #
        #     counter += 1
        while counter < min_samples_num:
            for f in files_in_folder:
                if counter >= min_samples_num:
                    break
                old_file = actual_folder + f
                x = cv2.imread(old_file)
                #img = load_img(old_file)
                #x = img_to_array(img)
                # Reshape the input image
                x = x.reshape((1,) + x.shape)



                datagen = ImageDataGenerator(rotation_range=20,
                                            zoom_range=0.15,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.15,
                                             )
                for batch in datagen.flow(x, batch_size=1, ):
                    file = batch[0]
                    new_file_name = createFileName(destination_folder, counter)
                    cv2.imwrite(new_file_name, file)
                    break
                counter += 1





def isImageOutOfRoi(img_coords, h, top_roi_ratio, bottom_roi_ratio,threshold=0.2):
    img_top_coord = img_coords[1]
    img_bottom_coord = img_coords[3]
    max_top = int(h * top_roi_ratio)
    max_bottom = int(h - (h * bottom_roi_ratio))
    biased_top_of_image = int((img_bottom_coord - img_top_coord) * threshold) + img_top_coord
    biased_bottom_of_image = img_bottom_coord - int((img_bottom_coord - img_top_coord) * threshold)

    #Ha a kép alsó része, a felső roi hátárán vagy az fölött van, akkor minden képpen kiesik a roiból
    if img_bottom_coord <= max_top:
        return True
    #Ha a kép felső része, az alsó roi hátárán vagy az alatt van, akkor minden képpen kiesik a roiból
    if img_top_coord >= max_bottom:
        return True
    #Ha a kép felső része és annak eltolt felső része is, kívül esik a felső roi határból, akkor kiesik a roiból
    if img_top_coord < max_top and biased_top_of_image < max_top:
        return True
    #Ha a kép alsó része és annak eltolt alsó része is, kívül esik az alsó roi határból, akkor kiesik a roiból
    if img_bottom_coord > max_bottom and biased_bottom_of_image > max_bottom:
        return True
    return False

def isImageContainsGroundTrouth(img_coords, annotation_list, threshold=0.3):
    for ann in annotation_list:
        val = IOU(img_coords, (ann.top_left[0], ann.top_left[1], ann.bottom_right[0], ann.bottom_right[1]))
        if val >= threshold:
            print(val)
            return True
    return False

def createFileName(prefix, counter, number_of_zeros=5, extension=".ppm"):
    result = prefix

    num = number_of_zeros - len(str(counter))

    for i in range(num):
        result += "0"
    result += str(counter) + extension
    return result

