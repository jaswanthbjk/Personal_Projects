Packages Used:

	1. OpenCV
	2. NumPy
	3. Matplotlib


Aim: The main aim of this projects is to detect the lanes on the road and mark them using the image processing and computer vision techniques.

Process:

My approach is to divide the project into various problems and try to solve them one by one.

The division is as follows

1. Loading the image and a video along with reading each frame of the video.

2. Process the image for

   1. Convert the image into a Grayscale image for white line marking detection and into HSV image for yellow marks detection.
   2. Detection the white and yellow patches and commonize the patches into a single patch by performing Bitwise operations.

3. Apply gaussian blur to reduce the amount of noise in the image and apply Edge detection to determine the line markings as lines.

4. Apply Hough transform to determine the lines present in the image inorder to mark them as a one continuous line without breaking.

5. Draw the lines as per the information obtained from the previous step and superimpose it onto the original image/frame that is read.

The rocess is executed as per the above plan and the result is achieved as expected.
