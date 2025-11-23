import os

def get_extension(path:str):
    _, format = os.path.splitext(path)
    return format