# encoding: utf-8
__author__ = 'Jonas Teuwen'
import json
import logging
import os
import pickle
import shutil
import subprocess
import zipfile

logger = logging.getLogger(__name__)


def save_object(python_object, filename):
    """
    Save a python object by pickling it.
    Parameters
    ----------
    python_object: object
    filename: string
    """
    filename = os.path.abspath(filename)
    with open(filename, 'wb') as f:
        pickle.dump(python_object, f, pickle.HIGHEST_PROTOCOL)


def read_object(filename):
    """
    Read a python object by unpickling it.
    Parameters
    ----------
    filename: string
    Returns
    -------
    Python object from pickle
    """
    filename = os.path.abspath(filename)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def write_json(filename, input_dict, indent=4):
    """Writes a dictionary to a json file.
    """
    json.dump(input_dict, open(filename, 'w'),
              sort_keys=True, indent=indent)


def read_json(filename):
    """Reads a json file to dictionary"""
    json_dict = json.load(open(filename, 'r'))
    return json_dict


def read_list(filename):
    """Reads file with caseids, separated by line.
    """
    f = open(filename, 'r')
    ids = []
    for line in f:
        if (line != '\n') and (line.strip()[0] != '#'):  # otherwise it is a comment or empty
            ids.append(line.strip())
    f.close()

    return ids


def write_list(filename, input_list, append=False):
    """Reads a list of strings and writes the list line by line to a text file."""
    mode = 'a' if append else 'w'
    with open(filename, mode) as f:
        for line in input_list:
            f.write(line.strip() + '\n')


def link_data(source_path, target_path, copy=False):
    """
    Link from source path to target path. Can be either copy or symbolic linking. Supports files and folders.
    If the symbolic link exists, then it is verified if the link is correct, otherwise an IOError is returned
    Parameters
    ----------
    source_path : str
    target_path : str
    copy : bool
    Returns
    -------
    None
    """
    if not copy:
        try:
            os.symlink(source_path, target_path, target_is_directory=os.path.isdir(source_path))
        except FileExistsError:
            if os.path.normpath(os.path.realpath(target_path)) == os.path.normpath(source_path):
                logger.info(f'Symlink from {source_path} to {target_path} already exists.')
            else:
                raise IOError(f'Symlink to {target_path}, but does not refer to {source_path}.')

    else:
        if os.path.isdir(source_path):
            shutil.copytree(source_path, target_path)
        else:
            shutil.copyfile(source_path, target_path)


def unzip(source_filename, dest_dir):
    with zipfile.ZipFile(source_filename) as zf:
        zf.extractall(dest_dir)


def git_hash(dir_to_git):
    """
    Get the current git hash
    Returns
    -------
    The git hash, otherwise None.
    """
    try:
        ghash = subprocess.check_output(
            ['git', 'rev-list', '-1', 'HEAD', './'], cwd=dir_to_git).strip().decode('utf-8')
    except subprocess.CalledProcessError:
        ghash = None
    return ghash