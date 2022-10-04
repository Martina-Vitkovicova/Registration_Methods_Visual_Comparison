from kivy.properties import StringProperty

import Project_2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout


class MyWidget(BoxLayout):
    label_text = StringProperty("Selected files:")
    files = []

    def selected(self, filenames):
        self.files.append(filenames[-1])
        self.label_text += "\n"
        self.label_text += filenames[-1]

    def visualize(self, files):
        object = Project_2.import_obj(files)
        print(len(files) // 2)
        if len(files) % 2 == 0:
            Project_2.visualize(object, True)
        else:
            Project_2.visualize(object)


class ProjectApp(App):
    def build(self):
        return MyWidget()


if __name__ == '__main__':
    ProjectApp().run()


