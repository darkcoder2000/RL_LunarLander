
def correct_folder_name(folder_name):
    """
    Corrects the folder name if it is not valid.
    """
    invalid_characters = '<>:"/\|?* '

    for char in invalid_characters:
        folder_name = folder_name.replace(char, '')

    return folder_name

