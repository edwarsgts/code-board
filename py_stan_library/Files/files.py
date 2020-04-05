# Files , csv files
from pathlib import Path
from time import ctime
from zipfile import ZipFile
import shutil
import csv
import json

#  methods:
#  exist()
#  is_file()
#  is_dir()
#  name
#  stem
#  suffix
#  parent
#  with_name("file.txt")
#  with_suffix(".txt")


path = Path()
print(path.name)
# print(path.name)
# print(path.suffix)
# print(path.absolute())

# Working with directories
# mkdir(), rmdir(), rename("new_name")
# iterdir() --> generator object

# print(path.is_dir())
# paths = [p for p in path.iterdir() if p.is_dir()]
# print(paths)

# PosixPath
# We cannot search by a pattern
# Cannot search recursively

# glob
# pyfile = [p for p in path.glob("*.py")]
# print(pyfile)

# to search recursively ( for children etc.)
# pyfile = [p for p in path.rglob("*.py")]
# print(pyfile)

# Working with files
# unlink() deletes
# stat() --> returns st_size=bytes, last access/mod/create time
# print(ctime(path.stat().st_ctime))

# read_bytes
# read_text
# this is better than opening a file (need to rmb to close)
#  with open () as var_name:

# write_text
# write_bytes

# not ideal for copying file
# use shutil.copy(source, target)

# Working with zip files
# throwing everything in your folder into a zip file
# with ZipFile("files.zip", "w") as zip:
#   for path in Path("").rglob("*.*"):
#       zip.write(path)

# Working with CSV files
# Writing csv file
# with open("path", "w") as file:
#   writer = csv.writer(file)
#   writer.writerow(["trans_id","product_id","price"])
#   writer.writerow([1000,1,5])
#   writer.writerow([1000,1,5])

# Reading csv file
# open file:
#   reader = csv.reader(file)
#   print(list(reader)) #moves pointer to the EOF
# Note: everything is a string
#   for row in reader:
#       print(row)

# combine file, summarizing file

