import torch
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import csv
import math
import sys

################## All classes and methods that we need ########################

"""
This class contains all the necessary informations about a droplet

PARAMETERS:
- x_coord = is the x coordinate at time t
- num_cells = the number of cells contained in the droplet
- list_cells = contains the number of cells for each frame
- close = True -> if two cells are close to each other. It often happens
                  that if two cells are close to each other, we always only
                  count one cell. Therefore, if we detect in one of the frames
                  two cells which are close to each other, we do not take the
                  average of the number of cells as final number, but the number
                  of cells, that we detected at that moment. Only exeption, if the
                  average is higher.
          False -> otherwise

METHODS:

- change_x_coord :
    This method checks if the droplet moves to the right. If it 
    moves to the right, we update the x coordinate.

    PARAMETERS : new_x_coord = the new coordinate of the droplet
    RETURN : True = if the droplet moves to the right (x_coord < new_x_coord)
             False = otherwise

- add_cells:
    This method adds the number of cells from the current frame to the list_cells

    PARAMETERS : number_of_cells = the number of cells in the droplet
    RETURN : 

- return_number_of_cells_
    This method takes the average of the list_cells. If the average is higher as .4, we
    round up, otherwise we round down.

    PARAMETERS : 
    RETURN : the predicted number of cells in this droplet

"""
class Droplet:

    close = False
    num_cells = 0

    def __init__(self, x_coord, num_cells):
        self.x_coord = x_coord
        self.list_cells = []
        self.list_cells.append(num_cells)

    def change_x_coord(self, new_x_coord):
        #We check if the droplet moved to the right
        #or if it is a new droplet
        if (new_x_coord > self.x_coord):
            self.x_coord = new_x_coord
            return True
        else:
            return False

    def add_cells(self, number_of_cells, close):
        if (close):
            self.close = True
            self.num_cells = number_of_cells
            self.list_cells.append(number_of_cells)
        else:
            self.list_cells.append(number_of_cells)
        
    def return_number_of_cells(self):
        if(self.close):
            average = sum(self.list_cells) / len(self.list_cells)
            # if the average is > 0.4, we round up
            final_number_of_cells = int(average + 0.6)

            if(self.num_cells > final_number_of_cells):
                return self.num_cells
            else:
                return final_number_of_cells
        else:
            average = sum(self.list_cells) / len(self.list_cells)
            # if the average is > 0.4, we round up
            final_number_of_cells = int(average + 0.6)
            return final_number_of_cells

"""
This method counts the number of cells inside the droplet. As in a frame, there might
be more than one droplet with multiples cells, we have to be sure to assign each cell
to the right droplet.

PARAMETERS : x_min = the x coordinates of the most left point of the droplet
             x_max = the x coordinates of the most right point of the droplet
             coord = contains all the information about each prediction of the model.
                     A list of lists in form of:
                     [ (x_min, y_min, x_max, y_max, confidence, label), ....]
             x_shape = the width of the frame
             y_shape = the hight of the frame
RETURN : counter = the number of cells in the droplet
"""
def count_cells_in_droplet(x_min, x_max, coord, x_shape, y_shape):
    n = len(labels)
    counter = 0
    close = False

    list_center = []

    for i in range(n):
        row = coord[i]
        x1, x2 = int(row[0]*x_shape), int(row[2]*x_shape)
        y1, y2 = int(row[1]*y_shape), int(row[3]*y_shape)
        center_x = (x2 + x1) / 2
        center_y = (y2 + y1) / 2
        if(row[5] == 0):
            continue
        else:
            list_center.append(center_x)
            list_center.append(center_y)
            if(center_x > x_min and center_x < x_max):
                counter+=1
    # We check if both cells are close to each other because it happen frequently that if two cells are too close 
    # to each other that we only detect one cell. Therefore, if we detect once two cells that are close to each other
    # we have to be sure that we take 2
    if(counter == 2):
        distance = math.sqrt( abs(list_center[0] - list_center[2])**2 + abs(list_center[1] - list_center[3])**2 )
        if(distance < 30):
            close = True

    return counter, close


"""
This method loads the yolov5s model.

PARAMETERS : conf_thres = the confidence threshold. If the confidence of an 
                          object detection is over the threshold, we define it
                          as detected.
             max_det = the maximum number of object in a frame
RETURN : model = the yolov5 model
"""
def load_model(conf_thres=0.5, max_det=10):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_640.pt', force_reload=True)
    model.conf = conf_thres
    model.max_det = max_det
    return model

"""
This method draws a box around a cell or droplet. Green box for a droplet and a blue box for a cell.
Furthermore, we also show the confidence of the object detection.

PARAMETERS : array_coord = contains all the information about the object which we detected
                           Has the following form : (x_min, y_min, x_max, y_max, confidence, label)
             img = the frame where we want to detect objects
             label = 0 if our object is a droplet
                     1 if our object is a cell
RETURN : img = the image with the drawn box and the corresponding confidence score
"""
def draw_box(array_coord, img, label):
    #x_shape = vidcap.get(3)
    #y_shape = vidcap.get(4)
    y_shape, x_shape, _ = img.shape
    x1, y1, x2, y2 = int(array_coord[0]*x_shape), int(array_coord[1]*y_shape), int(array_coord[2]*x_shape), int(array_coord[3]*y_shape)
    if(label == 0):
        bgr = (0, 255, 0)
        text = "Droplet " +  str(float("{:.2f}".format(array_coord[4])))
    else:
        bgr = (255, 0, 0)
        text = "Cell " +  str(float("{:.2f}".format(array_coord[4])))
        area = (x2 - x1) * (y2 - y1)

    img = np.array(img)
    cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
    y_middle = int((y2+y1)/2)
    #cv2.putText(img, text, (x2 + 5, y_middle), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)

    return img

"""
This method sorts a list of lists by the first element. In the new obtained list, the first
list is the object which is the most at the right, whereas the last list is the object which
is the most at the left.

PARAMETERS: list_to_sort = the list of lists which we want to sort. 
                           [(x_min, y_min, x_max, y_max, confidence, label),...]
RETURNS: list_to_sort = the sorted list

"""
def sort_list(list_to_sort):
    l = len(list_to_sort)
    for i in range(0, l):
        for j in range(0, l-i-1):
            if (list_to_sort[j][0] < list_to_sort[j + 1][0]):
                aux = list_to_sort[j]
                list_to_sort[j]= list_to_sort[j + 1]
                list_to_sort[j + 1]= aux
    return list_to_sort


################################# START OF THE CODE ########################################

# We load our yolov5s model. To load the model, you need to be connected
# to the internet in order to load the model from github.
model = load_model()


# We load the video
if(len(sys.argv) > 1):
    vidcap = cv2.VideoCapture(sys.argv[1])
    success,img = vidcap.read()
else:
    print("Please enter the path to the video")
    sys.exit()

"""
First element -> number of droplets without a cell
Second element -> number of droplet with one cell
Nth element -> number of droplet with N-1 cells
"""
list_of_cells = [0,0,0,0,0,0,0,0,0]

# Total number of droplets in the video
total_number_of_dropelts = 0

# List which contains everything that we want to write to
# the csv file later on.
list_write_to_file = []
list_write_to_file.append("0,0,0,0,0\n")

# saves the droplets from the previous frame
list_of_droplets_prev = []
length_droplets_prev = 0

# saves the droplets from the actuel frame
list_of_droplets_act = []
length_droplets_act = 0

frame_counter = 0

while success:
    # Our model predicts the object on the frame img with an image size of 640
    results = model(img, 640)

    labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    row = []
    list_rows = []
    n = len(labels)
    list_of_droplets_act = []
    for i in range(n):
        row = []
        row.append(cord[i][0])
        row.append(cord[i][1])
        row.append(cord[i][2])
        row.append(cord[i][3])
        row.append(cord[i][4])
        row.append(int(labels[i]))
        list_rows.append(row)

        if(labels[i] == 0):
            list_of_droplets_act.append(row)

    list_rows = sort_list(list_rows)

    # If no items were detected
    if(n == 0):
        row_to_add = "frame_" + str(frame_counter) + ",None,None,None,None,None,None\n"
        list_write_to_file.append(row_to_add)

    length_droplets_act = len(list_of_droplets_act)
    aux_droplet_counter = 0
    condition = length_droplets_act - len(list_of_droplets_prev)

    # condition = 0 -> same number of droplets
    # condition > 0 -> we have now more droplets than previously
    # condition < 0 -> we have now less droplets than previously

    """
    In the following we always assume that only one droplet at a time can leave or
    enter the screen. If two droplets enter or leave the screen at the same time, there
    is no guarantee that the code works correctly.
    """

    # we have more droplets than previously
    if(condition > 0):
        for i in range(n):
            row = list_rows[i]
            label = row[5]
            row_to_add = "frame_" + str(frame_counter) + "," + str(row[0]) + "," + str(row[1]) + "," + str(row[2]) + "," + str(row[3]) + "," + str(row[4])

            y_shape, x_shape, _ = img.shape
            img = draw_box(row, img, label)

            # we check if it is a droplet
            if(label == 0):
                row_to_add += "," + "droplet\n"
                list_write_to_file.append(row_to_add)

                x_min = int(row[0]*x_shape)
                x_max = int(row[2]*x_shape)
                x_center = (x_min +  x_max) / 2
                num_cells, close = count_cells_in_droplet(x_min, x_max, list_rows, x_shape, y_shape)

                aux_droplet_counter+=1
                if(aux_droplet_counter > length_droplets_prev):
                    new_droplet = Droplet(x_center, num_cells)
                    list_of_droplets_prev.append(new_droplet)
                else:
                    list_of_droplets_prev[aux_droplet_counter-1].change_x_coord(x_center)
                    list_of_droplets_prev[aux_droplet_counter-1].add_cells(num_cells, close)
            else:
                row_to_add += "," + "cell\n"
                list_write_to_file.append(row_to_add)
                continue

    flag = False
    move_right = False
    # we have the same number of droplets as previously
    if(condition == 0):
        for i in range(n):
            row = list_rows[i]
            label = row[5]
            row_to_add = "frame_" + str(frame_counter) + "," + str(row[0]) + "," + str(row[1]) + "," + str(row[2]) + "," + str(row[3]) + "," + str(row[4])

            y_shape, x_shape, _ = img.shape
            img = draw_box(row, img, label)

            # we check if it is a droplet
            if(label == 0):
                row_to_add += "," + "droplet\n"

                list_write_to_file.append(row_to_add)

                x_min = int(row[0]*x_shape)
                x_max = int(row[2]*x_shape)
                x_center = (x_min +  x_max) / 2
                num_cells, close = count_cells_in_droplet(x_min, x_max, list_rows, x_shape, y_shape)

                # we check if the droplet(s) is the same droplet(s) as previously or if one droplet left and a new droplet appeared

                # we compare the most right droplet from previously with the most right droplet from the actuel frame. If it moved right, we 
                # know that no droplet left the screen and no new droplet appeared.

                if(not flag):
                    flag = True
                    move_right = list_of_droplets_prev[0].change_x_coord(x_center)
                    if(move_right):
                        list_of_droplets_prev[0].add_cells(num_cells, close)
                    else:
                        old_droplet = list_of_droplets_prev.pop(0)
                        list_of_cells[old_droplet.return_number_of_cells()] += 1
                        total_number_of_dropelts += 1

                        if(len(list_of_droplets_act) > 1):
                            list_of_droplets_prev[aux_droplet_counter].change_x_coord(x_center)
                            list_of_droplets_prev[aux_droplet_counter].add_cells(num_cells, close)
                        else:
                            new_droplet = Droplet(x_center, num_cells)
                            list_of_droplets_prev.append(new_droplet)
                        
                else:
                    if(move_right):
                        aux_droplet_counter+=1
                        list_of_droplets_prev[aux_droplet_counter].change_x_coord(x_center)
                        list_of_droplets_prev[aux_droplet_counter].add_cells(num_cells, close)
                    else:
                        if(i == n-1):
                            new_droplet = Droplet(x_center, num_cells)
                            list_of_droplets_prev.append(new_droplet)
                        else:
                            aux_droplet_counter+=1
                            list_of_droplets_prev[aux_droplet_counter].change_x_coord(x_center)
                            list_of_droplets_prev[aux_droplet_counter].add_cells(num_cells, close)

            else:
                row_to_add += "," + "cell\n"
                list_write_to_file.append(row_to_add)
                continue

    # We have less droplets than previously
    if(condition < 0):
        # The most rigt droplet left the screen
        old_droplet = list_of_droplets_prev.pop(0)
        list_of_cells[old_droplet.return_number_of_cells()] += 1
        total_number_of_dropelts += 1

        for i in range(n):
            row = list_rows[i]
            label = row[5]
            row_to_add = "frame_" + str(frame_counter) + "," + str(row[0]) + "," + str(row[1]) + "," + str(row[2]) + "," + str(row[3]) + "," + str(row[4])

            y_shape, x_shape, _ = img.shape
            img = draw_box(row, img, label)

            # we check if it is a droplet
            if(label == 0):
                row_to_add += "," + "droplet\n"
                list_write_to_file.append(row_to_add)

                x_min = int(row[0]*x_shape)
                x_max = int(row[2]*x_shape)
                x_center = (x_min +  x_max) / 2
                num_cells, close = count_cells_in_droplet(x_min, x_max, list_rows, x_shape, y_shape)

                list_of_droplets_prev[aux_droplet_counter].change_x_coord(x_center)
                list_of_droplets_prev[aux_droplet_counter].add_cells(num_cells, close)  
                aux_droplet_counter += 1  

            else:
                row_to_add += "," + "cell\n"
                list_write_to_file.append(row_to_add)
                continue


    length_droplets_prev = len(list_of_droplets_prev)

    frame_counter+=1

    # Read new image from the video
    success,img = vidcap.read()


for droplet in list_of_droplets_prev:
    list_of_cells[droplet.return_number_of_cells()] += 1
    total_number_of_dropelts+=1

list_write_to_file[0] = str(list_of_cells[0]) + "," + str(list_of_cells[1]) + "," + str(list_of_cells[2]) + "," + str(list_of_cells[3]) + "," + str(list_of_cells[4]) + "\n"


# Open csv file and write all the information to it
f = open('results.csv', 'w')
for i in range(len(list_write_to_file)):
    f.write(list_write_to_file[i])
f.close()