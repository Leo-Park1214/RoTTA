
class SimpleClassMemory:
    def __init__(self, max_size=20):
        self.max_size = max_size
        self.class_window = []

    def update(self, class_id):
        self.class_window.append(class_id)
        if len(self.class_window) > self.max_size:
            self.class_window.pop(0)

    def get_memory(self):
        return self.class_window