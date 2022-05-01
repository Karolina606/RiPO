
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import configparser
from detector import Detector


class KivyCV(Image):
    """
    A class that update view for user.

    It extends kivy.uix.image.Image class.
    ...

    Methods
    -------
    update()
        Read frames form source and set it
        as a Image.texture.
    """

    def __init__(self, capture, fps, **kwargs):
        Image.__init__(self, **kwargs)
        self.capture = capture
        # set interval to read frames
        Clock.schedule_interval(self.update, 1.0 / fps)


    def update(self, dt):
        """Updates ui for user.
        
        Prints next frames from source (woth detected object)
        """

        ret, frame = self.capture.read()
        if ret:
            # use Detector class to detect object on frame
            frame = Detector.detect_crosswalk(frame)
            if frame is not None:
                # create texture from np.array
                buf = cv2.flip(frame, 0).tostring()
                image_texture = Texture.create(
                    size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                # display image from the texture
                self.texture = image_texture


class CrosswalkDetectionApp(App):
    """
    Main application class it extends kivy's App class.

    ...

    Methods
    -------
    build() -> KivyCV
        Runs when app starts

    on_stop() 
        Releases source of frames.
    """


    def build(self):
        """Runs when app starts.
        
        Get source of frmaes from config.ini file.
        Starts video capturing and show kivy Image component with KivyCV class.
        """
        config = configparser.ConfigParser()
        config.read('config.ini')

        print(config['input']['src'])
        if config['input']['src'] == 'camera':
            input = 0
        else:
            input = config['input']['src']

        try:
            self.capture = cv2.VideoCapture(input)
            my_camera = KivyCV(capture=self.capture, fps=60)
            return my_camera

        except Exception:
            print("Check if correct input in config.ini")

    def on_stop(self):
        self.capture.release()


if __name__ == '__main__':
    CrosswalkDetectionApp().run()
