"""
Output adjustment routines
"""

import os


def rewrite_docs(docs_dir):
    file_list = os.listdir(docs_dir)
    for file in file_list:
        lines = [line.rstrip('\n') for line in open(docs_dir + file)]
        rewrite_file(docs_dir + file, lines)
    return file_list


def rewrite_file(file_name_path, lines):
    file = open(file_name_path, 'w')
    for line in lines:
        if (line[0] == '[') and (line[-1] == ']'):
            line_temp = line[1:-1]
            file.write(line_temp + '\n')
        else:
            file.write(line + '\n')


def generate_csv(docs_dir, output_path, output_file_prefix):
    acc_file = open(output_path + output_file_prefix + 'acc.csv', 'w')
    time_file = open(output_path + output_file_prefix + 'time.csv', 'w')
    file_list = os.listdir(docs_dir)
    array_file_acc_data = []
    array_file_time_data = []
    for file in file_list:
        data_array = [extract_file_name(file)]
        lines = [line.rstrip('\n') for line in open(docs_dir + file)]
        for line in lines:
            data_array.append(line)
        if (file[-8:-4] == '.acc'):
            array_file_acc_data.append(data_array)
        elif (file[-8:-4] == 'time'):
            array_file_time_data.append(data_array)

    for array, out_file in [(array_file_acc_data, acc_file), (array_file_time_data, time_file)]:
        max_len = 0
        for data in array:
            if max_len < len(data):
                max_len = len(data)
        out_csv_lines = []
        for i in range(max_len):
            csv_line = ''
            for j in range(len(array)):
                if len(array[j]) - 1 < i:
                    val = ''
                else:
                    val = array[j][i]
                if j == len(array) - 1:
                    csv_line += (val)
                else:
                    csv_line += (val + ',')
            out_csv_lines.append(csv_line)
        for l in out_csv_lines:
            out_file.write(l + '\n')
    print 'DONE'


def extract_file_name(file):  # 'file.ext1.ext2'
    return file[0:file.index('.')]
