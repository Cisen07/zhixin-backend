import os
from werkzeug import secure_filename

basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(basedir, 'video')


def upload_file(file):
    print("get in the upload_file method")
    filename = secure_filename(file.filename)
    print("filename is ", filename)
    full_pathname = os.path.join(UPLOAD_FOLDER, filename)
    print("full_pathname is ", full_pathname)
    file.save(full_pathname)
