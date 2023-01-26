import zipfile
import conf as cfg
import os
import argparse

parser = argparse.ArgumentParser(prog='unzipy.py', description='this file extract file and save it in specific location')

parser.add_argument('--filename', '-fn', type=str, default=None)
parser.add_argument('--savepath', '-sp', type=str, default='./')

args = parser.parse_args()



with zipfile.ZipFile(os.path.join(cfg.paths[args.savepath], args.filename), 'r') as zip_ref:
    zip_ref.extractall(cfg.paths[args.savepath])