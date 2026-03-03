class staticproperty:
    def __init__(self, function):
        self.function = function

    def __get__(self, instance, owner):
        return self.function()
