import cv2
import numpy as np

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def cd_color_segmentation(img, template):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((left_bottom, left_top), (right_bottom, right_top)); the bounding boxes of the cones, unit in px
				For left box: (bottom_left, top_right)
				For right box: (bottom_right, top_left)
	"""
	########## YOUR CODE STARTS HERE ##########

	bounding_box = (((0,0),(0,0)), ((0,0),(0,0)))
	# image_print(img)
	# # Cut the image to the bottom half
	# img = img[img.shape[0]//2:, :, :]
	print(img.shape)
	image_print(img)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	
	# Define color range for white
	darker_white = np.array([75, 10, 178])
	brighter_white = np.array([90, 26, 255])
	
	# Create mask and apply morphological operations
	mask = cv2.inRange(hsv, darker_white, brighter_white)
	image_print(mask)

	kernel1 = np.ones((5, 5), np.uint8)
	kernel2 = np.ones((4, 4), np.uint8)
	mask = cv2.erode(mask, kernel2, iterations=1)
	mask = cv2.dilate(mask, kernel1, iterations=1)
	image_print(mask)

	# Find contours
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	# Initialize default bounding boxes (no detection)
	bounding_box = (((0, 0), (0, 0)), ((0, 0), (0, 0)))
	
	# Process contours if found
	if contours and len(contours) >= 2:
		# Sort contours by area and take the two largest
		largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
		
		# Get bounding rectangles for both contours
		rect1 = cv2.boundingRect(largest_contours[0])
		rect2 = cv2.boundingRect(largest_contours[1])
		
		# Ensure left contour is on the left side
		if rect1[0] > rect2[0]:
			rect1, rect2 = rect2, rect1
		
		# Extract coordinates for left box
		left_x, left_y, left_w, left_h = rect1
		left_bottom = (left_x, left_y + left_h)
		left_top = (left_x + left_w, left_y)
		
		# Extract coordinates for right box
		right_x, right_y, right_w, right_h = rect2
		right_bottom = (right_x + right_w, right_y + right_h)
		right_top = (right_x, right_y)
		
		# Set the bounding boxes
		bounding_box = ((left_bottom, left_top), (right_bottom, right_top))
		
		# Draw rectangles for visualization
		cv2.rectangle(img, (left_x, left_y), (left_x + left_w, left_y + left_h), (0, 0, 255), 2)
		cv2.rectangle(img, (right_x, right_y), (right_x + right_w, right_y + right_h), (0, 0, 255), 2)
		

		# Draw lines for visualization
		cv2.line(img, (left_x, left_y + left_h),(left_x + left_w, left_y), (255,0,0), 6)
		cv2.line(img, (right_x, right_y), (right_x + right_w, right_y + right_h), (255,0,0), 6)
	
		image_print(img)
	########### YOUR CODE ENDS HERE ###########
	
	print(bounding_box)
	return bounding_box



if __name__ == "__main__":
	pass

