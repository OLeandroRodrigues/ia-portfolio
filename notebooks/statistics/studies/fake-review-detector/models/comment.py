class comment:
    def __init__(self, author=None, text=""):
        self.author = author
        self.text = text

    def __str__(self):
        return f"{self.author or 'Anônimo'}: {self.text}"