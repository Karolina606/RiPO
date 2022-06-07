from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np
import configparser
from detector import Detector
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.camera import Camera
from kivy.lang import Builder
from kivy.core.audio import SoundLoader


# from android.permissions import request_permissions, Permission
 
 
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
        self.sound = SoundLoader.load("assets/mario.wav")

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
            frame, found = Detector.detect_crosswalk_as_one(frame)
            if frame is not None:
                if found:
                    self.sound.play()
                else:
                    self.sound.stop()

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
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        print(self.config['input']['src'])
        if self.config['input']['src'] == 'phone_camera':
            # only for tests with ip cam
            self.input = "http://192.168.140.254:8080/video"
        elif self.config['input']['src'] == 'camera':
            self.input = 0
        else:
            self.input = self.config['input']['src']

        try:
            self.capture = cv2.VideoCapture(self.input)
            self.my_camera = KivyCV(capture=self.capture, fps=60)
            layout = FloatLayout()
            button = Button(text="Camera / video input", 
                    size_hint=(1,0.12), 
                    pos_hint={'center_x': .5}, 
                    background_normal = 'assets/normal.png',
                    background_down = 'assets/down.png',
                    border = (30, 30, 30, 30),)
            button.bind(on_press=self.on_input_change)

            layout.add_widget(self.my_camera)
            layout.add_widget(button)
            return layout

        except Exception:
            print("Check if correct input in config.ini")

    def on_input_change(self, _):
        if self.input == 0:
            self.input = self.config['input']['src']
        else:
            self.input = 0
        self.my_camera.capture.release()
        self.my_camera.capture = cv2.VideoCapture(self.input)


    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
   CrosswalkDetectionApp().run()
