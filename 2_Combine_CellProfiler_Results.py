# Combine separate SQL outputs from CellProfiler and maps the gene names
# Usage:
# python Combine_CellProfiler_Results.py
# -p /path/to/CellProfiler/SQL/files
# -o /path/to/output/directory

import pandas as pd
import numpy as np
import re
import sys
import os
from time import sleep
from optparse import OptionParser

# Parameters to specify
parser = OptionParser()
parser.add_option('-p', '--path', dest='path', default='',
                  help='Path where the CellProfiler SQL files are held')
parser.add_option('-o', '--output-path', dest='output_path', default='',
                  help='Path to store output files')
parser.add_option('-m', '--gene', dest='gene',
                  default='EMA_MappingSheet.csv',
                  help='File which matches gene names to deletion strains')

(options, args) = parser.parse_args()

PATH_USE = options.path
MAPPING_SHEET = options.gene
PATH_OUT = options.output_path
if PATH_OUT == '':
    PATH_OUT = os.getcwd() + '/'
FILE_OUT = 'CPout_features_plate_list.txt'


def MergeImageAndObjectFile(f, Df_GeneNames, MetaDataKeep, ImageHeader, ObjectHeader):
    #Merges an image and object file from the SQL prefix f

    #Load image file
    infile_image = PATH_USE + f + '_Image.CSV'
    merge_on = ['Plate', 'FileName']
    Df_Image = pd.read_csv(infile_image, header=None, index_col=None)

    #Determine if this is a large or small file and attach header
    if Df_Image.shape[1] == len(ImageHeader):
        Df_Image.columns = ImageHeader
    else:
        sys.exit('File does not match either the standard or alt headers')

    if 'Image_PathName_GFP' in Df_Image.columns:
        image_paths = Df_Image['Image_PathName_GFP'].values.tolist()
    elif 'Image_PathName_PreGFP' in Df_Image.columns:
        image_paths = Df_Image['Image_PathName_PreGFP'].values.tolist()
    elif 'Image_PathName_UncorrGFP' in Df_Image.columns:
        image_paths = Df_Image['Image_PathName_UncorrGFP'].values.tolist()
    try:
        plate = [str(int(m.group(1))) for l in image_paths for m in [re.search("Plate([0-9]*)",l)]] # if m]
    except:
        plate = [os.path.basename(l) for l in image_paths]
        merge_on = ['FileName']

    if 'Image_FileName_GFP' in Df_Image.columns:
        files = [x[:-5] + '.flex' for x in Df_Image['Image_FileName_GFP'].values.tolist()]
    elif 'Image_FileName_PreGFP' in Df_Image.columns:
        files = [x[:-5] + '.flex' for x in Df_Image['Image_FileName_PreGFP'].values.tolist()]
    elif 'Image_FileName_UncorrGFP' in Df_Image.columns:
        files = [x[:-5] + '.flex' for x in Df_Image['Image_FileName_UncorrGFP'].values.tolist()]

    Df_Image['Plate'] = plate
    Df_Image['FileName'] = files
    Df_Image_Genes = pd.merge(Df_GeneNames,Df_Image,on=merge_on)
    assert Df_Image_Genes.shape[0] > 0, 'Failed to merge GeneName information on'
    #Reduce to features of interest
    Df_Image_Genes_Sub = Df_Image_Genes[[x for x in Df_Image_Genes.columns if x in MetaDataKeep]]

    #Load object file
    infile_object = PATH_USE + f + '_Object.CSV'
    Df_Object = pd.read_csv(infile_object, header = None,index_col = None)

    #Determine if this is a large or small file and attach header
    if Df_Object.shape[1] == len(ObjectHeader):
        Df_Object.columns = ObjectHeader
    else:
        sys.exit('File does not match either the standard or alt headers')

    #Merge image and object file
    Df_Full = pd.merge(Df_Image_Genes_Sub, Df_Object , on='ImageNumber')
    return Df_Full


def ParseSQLSETUP(infile):
    #Reads the SQL_SETUP.SQL file into two lists
    f = open(infile)
    r = {}
    r['image'] = []
    r['object'] = []
    switch = 0
    for line in f:
        line = line.rstrip('\n')
        if 'CREATE TABLE' in line:
            if 'Image' in line:
                switch = 1
                type = 'image'
            elif 'Object' in line:
                switch = 1
                type = 'object'
        elif 'PRIMARY KEY' in line:
            switch = 0
        elif switch == 1:
            val = line.split(' ')[0]
            val = val.replace(',','')
            r[type].append(val)
    return r['image'], r['object']


def WriteCurrent(Df, p, cols, to_analyze):
    outfile = p[0] + '_rawdata.csv'
    print(outfile)
    Df = Df[cols]
    Df.reset_index(drop=True)
    Df.to_csv(PATH_OUT + outfile, index=False)
    to_analyze.append(PATH_OUT + outfile)

    return to_analyze


def ReadOrWriteINFILE(name, path):
    # Check if SQL files are already merged or not
    for root, dirs, files in os.walk(path):
        if name in files:
            f = open(path + name, 'r')
            INFILE_LIST = [x.strip() for x in f.readlines()]
            return INFILE_LIST

    # Merge SQL files
    INFILE_LIST = PlatesToAnalyzeForOutlierDetection()
    np.savetxt(path + name, INFILE_LIST, fmt='%s')
    return INFILE_LIST


def PlatesToAnalyzeForOutlierDetection():
    #This function loads up the SQL_ and support files, merges the image and object, and writes the final table to OUTFILE

    # Headers
    ImageHeader, ObjectHeader = ParseSQLSETUP(PATH_USE + '/SQL_SETUP.SQL')

    # Genes
    Df_GeneNames = pd.read_csv(MAPPING_SHEET, index_col=None)
    if 'Plate' in Df_GeneNames:
        Df_GeneNames['Plate'] = Df_GeneNames['Plate'].astype(str)

    # MetaDataToKeep
    MetaDataKeep = ['Plate', 'ImageNumber', 'Image_Count_Cells', 'Image_Count_Nuclei', 'Image_FileName_GFP', 'Image_FileName_RFP']
    MetaDataKeep += Df_GeneNames.columns.values.tolist()

    # Create a list of the SQL files to look through
    files = os.listdir(PATH_USE)
    files_sql_prefixs = [m.group(1) for x in files for m in [re.search('(SQL_[0-9]*_[0-9]*)_',x)] if m]
    files_sql_prefixs = set(files_sql_prefixs)
    files_sql_prefixs = list(files_sql_prefixs)
    files_sql_prefixs = sorted(files_sql_prefixs, key=lambda item: int(item.split('_')[1]))

    Df_final = pd.DataFrame()
    plates_to_analyze = []
    cols_order_use = []
    print("files_sql_prefixs:", files_sql_prefixs)
    for f in files_sql_prefixs:
        Df_ObjectImage = MergeImageAndObjectFile(f, Df_GeneNames, MetaDataKeep, ImageHeader, ObjectHeader)
        cols_order_use = Df_ObjectImage.columns.values.tolist()
        Df_final = pd.concat([Df_final, Df_ObjectImage])

        for plate, df_plate in Df_final.groupby('Plate'):
            print("writing plate:", plate)
            plates_to_analyze = WriteCurrent(df_plate, [plate], cols_order_use, plates_to_analyze)
       
    return plates_to_analyze


if __name__ == '__main__':
    INFILE_LIST = ReadOrWriteINFILE(FILE_OUT, PATH_OUT)
