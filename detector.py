from ctypes import pointer
import cv2
import numpy as np
# import tensorflow.compat.v1 as tf
import sys

class Detector:

    # @staticmethod
    # def detect_with_tf(img):
    #     # Read in the image_data
    #     image_data = img

    #     # Loads label file, strips off carriage return
    #     label_lines = [line.rstrip() for line 
    #                     in tf.gfile.GFile("/home/karolina/Desktop/studejszyn_sem6/RiPO/tf_files1/retrained_labels.txt")]

    #     # Unpersists graph from file
    #     with tf.gfile.FastGFile("/home/karolina/Desktop/studejszyn_sem6/RiPO/tf_files1/retrained_graph.pb", 'rb') as f:
    #         graph_def = tf.GraphDef()
    #         graph_def.ParseFromString(f.read())
    #         _ = tf.import_graph_def(graph_def, name='')

    #     with tf.Session() as sess:
    #         # Feed the image_data as input to the graph and get first prediction
    #         softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            
    #         image_np_expanded = np.expand_dims(image_data, axis=0)
    #         predictions = sess.run(softmax_tensor, feed_dict={softmax_tensor: image_data})
            
    #         # Sort to show labels of first prediction in order of confidence
    #         top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            
    #         for node_id in top_k:
    #             human_string = label_lines[node_id]
    #             score = predictions[0][node_id]
    #             print('%s (score = %.5f)' % (human_string, score))

    
    """
    A class used to detect objects (crosswalks) in the image frame.

    ...

    Methods
    -------
    detect_crosswalk(img : numpy.array) -> numpy.array
        Returns image frame with object detected.
    """

    @staticmethod
    def detect_crosswalk(img):
        """ Returns image frame with object detected.

        Parameters
        ----------
        img : numpy.array
            Image frame

        Returns
        ----------
        numpy.array
            The same frame as given as parameter but with detected object (prints red ractangle around object)
        """

        # change colors to grayscale, add some blur, recognize contours and distinguish bright and dark colrs
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 6)
        ret, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

        # get all detected countorus (tree hierarchy)
        contours, hier = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        for countur in contours:
            # if the contour is not sufficiently large or too big, ignore it
            if  cv2.contourArea(countur) < 1000 or cv2.contourArea(countur) > 2000 :
                continue

            # get the min area rect
            rect = cv2.minAreaRect(countur)
            box = cv2.boxPoints(rect)

            # convert all coordinates floating point values to int
            box = np.int0(box)

            # draw a red 'nghien' rectangle
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

        return img


    @staticmethod
    def detect_crosswalk_only_center(img):
        """ Returns image frame with object detected.
        
        Parameters
        ----------
        img : numpy.array
            Image frame
        Returns
        ----------
        numpy.array
            The same frame as given as parameter but with detected object (prints red ractangle around object)
        """
        
        # Detector.detect_with_tf(img)
        # convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # apply a gaussian blur to the image
        blur = cv2.GaussianBlur(gray, (15,15), 6)
        # apply bit thresholding to the image to find white paint on black background
        ret, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)

        # define the contour search parameters and find the contours in the image
        min_area = 500
        max_area = 1000
        min_x = int(0.1 * img.shape[1])
        max_x = int(0.9 * img.shape[1])
        min_y = int(0.25 * img.shape[0])
        max_y = int(0.75 * img.shape[0])
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # loop through the contours and find the largest contour
        for c in contours:
            area = cv2.contourArea(c)
            if area > min_area and area < max_area:
                x, y, w, h = cv2.boundingRect(c)
                if x > min_x and x + w < max_x and y > min_y and y + h < max_y:
                    # draw a red rectangle around the contour and write the area of the contour
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img, str(area), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # draw area of interest
        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

        return img


    @staticmethod
    def detect_crosswalk_as_one(img):
        """ Returns image frame with object detected.
        
        Parameters
        ----------
        img : numpy.array
            Image frame
        Returns
        ----------
        numpy.array
            The same frame as given as parameter but with detected object (prints red ractangle around object)
        """


        kernel_Ero = np.ones((3,1),np.uint8)
        kernel_Dia = np.ones((3,5),np.uint8)

        copy_img = img.copy()
        copy_img = cv2.resize(copy_img,(1600,800))


        min_area = 800
        max_area = 1500
        min_x = int(0.1 * copy_img.shape[1])
        max_x = int(0.9 * copy_img.shape[1])
        min_y = int(0.25 * copy_img.shape[0])
        max_y = int(0.75 * copy_img.shape[0])
        imgGray = cv2.cvtColor(copy_img,cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray,(5,5),0)
        ret,thresh = cv2.threshold(imgBlur,180,255,cv2.THRESH_BINARY)
        imgEro = cv2.erode(thresh,kernel_Ero,iterations=2)
        imgDia = cv2.dilate(imgEro,kernel_Dia,iterations=4)

        contouts,hie = cv2.findContours(imgDia,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cnt = contouts

        for i in cnt:
            area = cv2.contourArea(i)
            if area > min_area and area < max_area:
                x,y,w,h = cv2.boundingRect(i)
                if x > min_x and x + w < max_x and y > min_y and y + h < max_y:
                    w += int(0.3 * w)
                    h += int(0.3 * h) 
                    # cv2.drawContours(img, i, -1, (0, 255, 0), thickness=2)

                    points = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
                    cv2.drawContours(copy_img, [points], -1, (255, 255, 255), thickness=20)
                    cv2.fillPoly(copy_img, [points], color=(255, 255, 255))


        # Once again
        imgGray = cv2.cvtColor(copy_img,cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray,(5,5),0)
        ret,thresh = cv2.threshold(imgBlur,220,255,cv2.THRESH_BINARY)
        imgEro = cv2.erode(thresh,kernel_Ero,iterations=2)
        imgDia = cv2.dilate(imgEro,kernel_Dia,iterations=4)

        contouts,hie = cv2.findContours(imgDia,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cnt = contouts

        img = cv2.resize(img,(1600,800))
        min_area = 15000
        max_area = 90000

        found = False
        for i in cnt:
            area = cv2.contourArea(i)
            if area > min_area and area < max_area:
                x,y,w,h = cv2.boundingRect(i)
                if x > min_x and x + w < max_x and y > min_y and y + h < max_y:
                    if w > 2.4 * h:
                        cv2.drawContours(img, i, -1, (0, 0, 255), thickness=2)
                        found = True
        # draw area of interest
        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

            
        return img, found
