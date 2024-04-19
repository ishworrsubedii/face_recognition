"""
Created By: ishwor subedi
Date: 2024-03-21
"""

import os


def rename_files(directory, old_name, new_name):
    for filename in os.listdir(directory):
        if old_name in filename:
            parts = filename.split('.')
            new_filename = f"{new_name}.0.{parts[2]}.jpg"
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))


if __name__ == '__main__':
    # Usage
    rename_files('resources/dataset', 'User', 'yash')
