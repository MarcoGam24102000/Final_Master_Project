def write_number_tests_to_file(numberTests):
    with open('temp_numberTests.txt', 'a') as file:
        file.write(str(numberTests))

def read_number_tests_from_file():
    with open('temp_numberTests.txt', 'r') as file:
        numberTests = int(file.read())

    # Delete file after reading
    import os
    os.remove('temp_numberTests.txt')

    return numberTests
