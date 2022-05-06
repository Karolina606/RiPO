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

        If the argument `sound` isn't passed in, the default Animal
        sound is used.

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
        min_area = 80
        max_area = 200
        min_x = 100
        max_x = img.shape[1] - 100
        min_y = 280
        max_y = img.shape[0] - 200
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
