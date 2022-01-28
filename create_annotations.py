from cytomine import Cytomine
from cytomine.models import *
from cytomine.models.image import SliceInstanceCollection
from shapely import wkt
from shapely.affinity import affine_transform
import cv2
from os.path import exists
import os
import matplotlib.pyplot as plt

"""
CREATION OF THE DATASET

1. In the first part, we split the labels which contains the information about the
   annotations inside a given image. and save them in either a training, validation or test set.
   The training set contains 70% of out dataset, the validation set 20% and the test set contains 10%.
2. In the second part, for each label we save the corresponding image. 
   (image and text file must have the same name).
3. Folder structure for the images : train_data/images
4. Folder structure for the labels : train_data/labels
"""

"""
SAVE IMAGES
"""
number_of_groups = 20

for x in range(number_of_groups):
	img_counter = 0
	if(x+1 < 10):
		name = "0" + str(x+1)
	else:
		name = str(x+1)

	path = "images/CV2021_GROUP{}/group{}.mp4".format(name,x+1)
	vid = cv2.VideoCapture(path)

	print("Loading images from : CV2021_GROUP{}".format(name))

	while(True):
		# Capture the video frame by frame
		success, frame = vid.read()

		if success == True:

			if img_counter > 400 and img_counter < 1002:
				img_name = "train_data/images/train/CV2021_GROUP{}_opencv_frame_{}.jpg".format(name, img_counter)
				cv2.imwrite(img_name, frame)
			elif img_counter > 1002 and img_counter < 1174:
				img_name = "train_data/images/val/CV2021_GROUP{}_opencv_frame_{}.jpg".format(name, img_counter)
				cv2.imwrite(img_name, frame)
			elif img_counter > 1174 and img_counter < 1261:
				img_name = "train_data/images/test/CV2021_GROUP{}_opencv_frame_{}.jpg".format(name, img_counter)
				cv2.imwrite(img_name, frame)
			else:
				img_counter+=1
				continue
			img_counter+=1
		else:
			break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

"""
SAVE ANNOTATIONS
"""

host = "https://learn.cytomine.be"
public_key = '76608821-e6c1-4393-9f73-2d9af0c7e306'
private_key = '03cfc269-3703-421e-a9db-62c79fe0c519'
conn = Cytomine.connect(host, public_key, private_key)

# ... Connect to Cytomine (same as previously) ...
projects = ProjectCollection().fetch()

for project in projects:

	print('### GROUP: {} ###'.format(project.name))

	images = ImageInstanceCollection().fetch_with_filter("project", project.id)
	terms = TermCollection().fetch_with_filter("project", project.id)
	slices = SliceInstanceCollection().fetch_with_filter("imageinstance", images[0].id)

	annotations = AnnotationCollection()
	annotations.showWKT = True
	annotations.showMeta = True
	annotations.showTerm = True
	annotations.showTrack = True
	annotations.showImage = True
	annotations.showSlice = True
	annotations.project = project.id

	annotations.fetch()

	for annot in annotations:

		#print("ID: {} | Image: {} | Project: {} | Terms: {} | Track: {} | Slice: {}".format(
		#	annot.id, annot.image, annot.project, terms.find_by_attribute("id", annot.term[0]), annot.track, annot.time))

		geometry = wkt.loads(annot.location)
		image = images.find_by_attribute("id", annot.image)
    
		geometry_opencv = affine_transform(geometry, [1, 0, 0, -1, 0, image.height])

		# We change the annotations in the needed format for yolov5
		# Format : class_id ¦ center_x ¦ center_y ¦ width ¦ height
		# class_id : 0 if Cell and 1 if Droplet
		# center_x and center_y : normalized coordinations of the center
		# width and height : normalized width and height of the annotation
		try:
			list_coord = list(geometry_opencv.exterior.coords)
		except:
			continue 
		
		x_left = abs(list_coord[1][0])
		x_right = abs(list_coord[0][0])
		y_high = abs(list_coord[0][1])
		y_low = abs(list_coord[2][1])

		center_x = ((x_right + x_left) / 2) / image.width
		center_y = ((y_high + y_low) / 2) / image.height

		width = abs((x_right - x_left) / image.width)
		height = abs((y_high - y_low) / image.height)

		# sometimes the height of the annotation is bigger than the image, 
		# then we change it to 1
		if(height > 1):
			height = 1

		if(width > 1):
			width = 1


		
		if annot.time > 400 and annot.time < 1002:
			path_f = "train_data/labels/train/{}_opencv_frame_{}.txt".format(project.name, annot.time)
			img_name = "train_data/images/train/{}_opencv_frame_{}.jpg".format(project.name, annot.time)
		elif annot.time > 1002 and annot.time < 1174:
			path_f = "train_data/labels/val/{}_opencv_frame_{}.txt".format(project.name, annot.time)
			img_name = "train_data/images/val/{}_opencv_frame_{}.jpg".format(project.name, annot.time)
		elif annot.time > 1174 and annot.time < 1261:
			path_f = "train_data/labels/test/{}_opencv_frame_{}.txt".format(project.name, annot.time)
			img_name = "train_data/images/test/{}_opencv_frame_{}.jpg".format(project.name, annot.time)
			
		else:
			continue
	

		if(os.path.exists(img_name)):
			if(os.path.exists(path_f)):
				file = open(path_f , 'a')
				file.write('\n')
				if(annot.term[0] == 11230348):
					file.write("0 " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height))
				else:
					file.write("1 " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height))
			else:
				file = open(path_f , 'w')
				if(annot.term[0] == 11230348):
					file.write("0 " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height))
				else:
					file.write("1 " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height))