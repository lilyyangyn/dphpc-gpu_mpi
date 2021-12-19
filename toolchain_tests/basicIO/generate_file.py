def gen_file(path, size):
    data = ''
    for i in range(size):
        data += '1'
    data += '\n'
    f = open(path, 'w')
    f.write(data)


if __name__ == '__main__':
    gen_file("./testfile.txt", 1024*1024)

