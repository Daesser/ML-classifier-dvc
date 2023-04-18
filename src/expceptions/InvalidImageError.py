class InvalidImageError(Exception):
    def __init__(self, image, message="invalid image"):
        self.image = image
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.image} is a {self.message}'
