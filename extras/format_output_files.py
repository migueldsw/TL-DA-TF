"""
Output adjustment routines
"""

import os


def rewrite_docs(docs_dir):
    file_list = os.listdir(docs_dir)
    for f in file_list:
        lines = [line.rstrip('\n') for line in open(docs_dir + f)]
        rewrite_file(docs_dir + f, lines)
    print '%d files modified.' % len(file_list)


def rewrite_file(file_name_path, lines):
    f = open(file_name_path, 'w')
    for line in lines:
        if (line[0] == '[') and (line[-1] == ']'):
            line_temp = line[1:-1]
            f.write(line_temp + '\n')
        else:
            f.write(line + '\n')


def generate_csv(docs_dir, output_path, output_file_prefix):
    acc_file = open(output_path + output_file_prefix + 'acc.csv', 'w')
    time_file = open(output_path + output_file_prefix + 'time.csv', 'w')
    file_list = os.listdir(docs_dir)
    array_file_acc_data = []
    array_file_time_data = []
    for f in file_list:
        data_array = [extract_file_name(f)]
        lines = [line.rstrip('\n') for line in open(docs_dir + f)]
        for line in lines:
            data_array.append(line)
        if f[-8:-4] == '.acc':
            array_file_acc_data.append(data_array)
        elif f[-8:-4] == 'time':
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
                    csv_line += val
                else:
                    csv_line += (val + ',')
            out_csv_lines.append(csv_line)
        for l in out_csv_lines:
            out_file.write(l + '\n')
    print 'DONE'


def extract_file_name(file_name):  # 'file.ext1.ext2'
    return file_name[0:file_name.index('.')]


def cfl(docs_dir):#count files lines
    file_list = os.listdir(docs_dir)
    lines_count_list = []
    for f in file_list:
        lines = [line.rstrip('\n') for line in open(docs_dir + f)]
        lines_count_list.append(len(lines))
    print '%d files, avg lines= %f, (min,max)=(%d,%d)' % (len(file_list), float(sum(lines_count_list))/len(lines_count_list),min(lines_count_list),max(lines_count_list))