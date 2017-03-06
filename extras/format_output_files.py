"""
Output adjustment routines
"""

import os

def rewrite_docs(docs_dir):
    file_list = os.listdir(docs_dir)
    for file in file_list:
        lines = [line.rstrip('\n') for line in open(docs_dir+file)]
        rewrite_file(docs_dir+file, lines)
    return file_list

def rewrite_file(file_name_path, lines):
    file = open(file_name_path,'w')
    for line in lines:
        if (line[0]=='[') and (line[-1]==']'):
            line_temp = line[1:-1]
            file.write(line_temp+'\n')
        else:
            file.write(line+'\n')

