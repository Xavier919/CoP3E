import csv
from Bio import SeqIO
import pyfaidx
import pickle
import numpy as np
import torch
import numpy as np


def trx_orfs(ensembl_trx, tsv_file):
    trx_orfs = dict()
    with open(tsv_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for n, row in enumerate(reader):
            if n == 0:
                continue
            if n == 1:
                cols = row
                continue
            line = dict(zip(cols, row))
            if "ENST" not in line["transcript accession"]:
                continue
            if not any(x in line["protein accession numbers"] for x in ["IP_", "ENSP", "II_"]):
                continue
            trx = line["transcript accession"].split(".")[0]
            altprot = line["protein accession numbers"].split(".")[0]
            seq = ensembl_trx[trx]['sequence']
            start, stop = int(line['start transcript coordinates'])-1, int(line['stop transcript coordinates'])-1
            start_codon, stop_codon = seq[start:start+3], seq[stop-3:stop]
            frame = int(line['frame'])
            chromosome = line['chr']
            altprots = dict()
            ##############QUALITY CONTROL#######################
            #if frame == 0:
            #    continue
            #if start == 0 and start_codon not in ['ATG']:
            #    continue
            #if stop_codon not in ['TAA','TAG','TGA']:
            #    continue
            ###################################################
            if trx not in trx_orfs:
                altprots[altprot] = {'MS':int(line["MS score"]),
                                    'TE':int(line["TE score"]),
                                    'unique_pept':0,
                                    'domains':int(line["Domains"]),
                                    'start':start,
                                    'start_codon':start_codon,
                                    'stop':stop,
                                    'stop_codon':stop_codon,
                                    'chromosome':chromosome,
                                    'ORF_length':stop-start,
                                    'biotype':ensembl_trx[trx]['Biotype_level1'],
                                    'frame':frame,
                                    'gene_name':ensembl_trx[trx]['symbol']
                                    }
                trx_orfs[trx] = altprots
            else:
                trx_orfs[trx][altprot] = {'MS':int(line["MS score"]),
                                        'TE':int(line["TE score"]),
                                        'unique_pept':0,
                                        'domains':int(line["Domains"]),
                                        'start':start,
                                        'start_codon':start_codon,
                                        'stop':stop,
                                        'stop_codon':stop_codon,
                                        'chromosome':chromosome,
                                        'ORF_length':int(line['stop transcript coordinates'])-int(line['start transcript coordinates']),
                                        'biotype':ensembl_trx[trx]['Biotype_level1'],
                                        'frame':frame,
                                        'gene_name':ensembl_trx[trx]['symbol']
                                        }
    for trx in ensembl_trx.keys():
        if trx not in trx_orfs:
            altprots = dict()
            trx_orfs[trx] = altprots
    return trx_orfs