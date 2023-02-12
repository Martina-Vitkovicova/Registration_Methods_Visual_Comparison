import registration_methods
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
from kivy.core.window import Window


Window.clearcolor = (0.1, 0.15, 0.2, 1)


class MyWidget(BoxLayout):
    key_label_text = ObjectProperty()
    other_label_text = ObjectProperty()
    key_file = []
    other_files = []
    method = ""

    def selected(self, filenames, id):
        if id == "key_fch":
            self.key_file = filenames
            self.key_label_text = ''.join([str(elem) + "\n" for elem in filenames])
        else:
            self.other_files = filenames
            self.other_label_text = ''.join([str(elem) + "\n" for elem in filenames])

    def save_selection(self, selection):
        self.method = selection

    def visualize(self):
        if not self.method:
            print("Please select the aligning method")
            return

        if not self.key_file or not self.other_files:
            print("Wrong file selection")
            return

        Project_2.visualization_method(self.method, self.key_file, [self.other_files[0]], self.other_files[1:])


class CustomDropDown(BoxLayout):
    pass


class Project2App(App):
    def build(self):
        return MyWidget()


if __name__ == '__main__':
    Project2App().run()
