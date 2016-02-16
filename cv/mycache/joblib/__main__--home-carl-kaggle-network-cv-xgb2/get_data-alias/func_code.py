# first line: 68
@mem.cache
def get_data(name):
    data = load_svmlight_file(name)
    return data[0], data[1]
