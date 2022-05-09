import cv2
import numpy as np


class Detector:
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
    # convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # apply a gaussian blur to the image
        blur = cv2.GaussianBlur(gray, (5,5), 0)
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
