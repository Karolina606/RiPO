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
