import sys
import cv2

def get_data(input_path):
	"""
    Annotation file is simple csv files formed from xml file generated by lableimg,
    This module be Parsing the data from annotation file
	file_path,x1,y1,x2,y2,class_name are to be extracted where 
    file_path is the absolute file path for this image
    (x1,y1) and (x2,y2) represent the top left and bottom right real coordinates of the original image
    class_name is the class name of the current bounding box.

	Args:
		input_path: The path of the csv/txt file (annotation file)

	Returns:
		
        all_data: list(filepath, width, height, list(bboxes))
		
        classes_count: dict{key:class_name, value:count_num} 
			e.g. {'Apple-Scab-Leaf': 158, 'Apple-leaf': 236, 'Apple-rust-leaf': 168, 'Bell_pepper-leaf': 312, 'Bell_pepper-leaf-spot': 249, 'Blueberry-leaf': 823,
		
        class_mapping: dict{key:class_name, value: idx}
			e.g. {'Apple-Scab-Leaf': 0, 'Apple-leaf': 1, 'Apple-rust-leaf': 2...}

	"""
	found_bg = False        #initially set backgrounds as zero, we'll count as we'll pass it into model

    #We'll declare variables 
	all_imgs = {}
	classes_count = {}
	class_mapping = {}

	i = 1
	
	with open(input_path,'r') as f:

		print('Parsing annotation files')

		for line in f:

			# Print process
			sys.stdout.write('\r'+'idx=' + str(i))
			i += 1

			line_split = line.strip().split(',')

			# Make sure the info saved in annotation file matching the format (path_filename, x1, y1, x2, y2, class_name)
			#	One path_filename might has several classes (class_name)
			#	x1, y1, x2, y2 are the pixel value of the origial image, not the ratio value
			#	(x1, y1) top left coordinates; (x2, y2) bottom right coordinates
			
			(filename,x1,y1,x2,y2,class_name) = (line_split[0][1:],line_split[1],line_split[2],line_split[3],line_split[4],line_split[-1][:-1])#(filename,x1,y1,x2,y2,class_name) = line_split

			if class_name not in classes_count:
				classes_count[class_name] = 1
			else:
				classes_count[class_name] += 1

			if class_name not in class_mapping:
				if class_name == 'bg' and found_bg == False:
					print("This is background")
					found_bg = True
				class_mapping[class_name] = len(class_mapping)

			if filename not in all_imgs:
				all_imgs[filename] = {}
				
				img = cv2.imread(filename)
				(rows,cols) = img.shape[:2]                 #make sure the image is properly loaded else it will throw error
				all_imgs[filename]['filepath'] = filename
				all_imgs[filename]['width'] = cols
				all_imgs[filename]['height'] = rows
				all_imgs[filename]['bboxes'] = []
				

			all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})


		all_data = []
		for key in all_imgs:
			all_data.append(all_imgs[key])
		
		#makesure the backgroundclass is the last one in this list
		if found_bg:
			if class_mapping['bg'] != len(class_mapping) - 1:
				key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
				val_to_switch = class_mapping['bg']
				class_mapping['bg'] = len(class_mapping) - 1
				class_mapping[key_to_switch] = val_to_switch
		
		return all_data, classes_count, class_mapping