import os

def print_models(model_folder='models'):
    for file_name in os.listdir(model_folder):
        if file_name.endswith('.pth'):
            print(file_name)