import pickle
from collections import Counter
import numpy as np
import copy
from sklearn import preprocessing
import matplotlib.pyplot as plt
import statistics
import scipy.stats
from scipy.stats import chisquare
from scipy.stats import chi2_contingency
import pandas as pd
import seaborn as sns

class FeatureCalculator:
    def __init__(self):
        
        self.nucleotides = ['A', 'C', 'T', 'G']
        self.dinucleotides = ['AT', 'AC', 'AG', 'AA', 'TA', 'TC', 'TG', 'TT', 'GT', 'GA', 'GC', 'GG', 'CA', 'CT', 'CG', 'CC']
        self.codons = ['ATA', 'ATC', 'ATT', 'ATG', 'ACA', 'ACC', 'ACG', 'ACT', 'AAC', 'AAT', 'AAA', 'AAG', 'AGC', 'AGT', 'AGA', 'AGG', 'CTA', 'CTC', 'CTG', 'CTT', 'CCA', 'CCC', 'CCG', 'CCT', 'CAC', 'CAT', 'CAA', 'CAG', 'CGA', 'CGC', 'CGG', 'CGT', 'GTA', 'GTC', 'GTG', 'GTT', 'GCA', 'GCC', 'GCG', 'GCT', 'GAC', 'GAT', 'GAA', 'GAG', 'GGA', 'GGC', 'GGG', 'GGT', 'TCA', 'TCC', 'TCG', 'TCT', 'TTC', 'TTT', 'TTA', 'TTG', 'TAC', 'TAT', 'TAA', 'TAG', 'TGC', 'TGT', 'TGA', 'TGG']
        self.amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '*']
        self.codon_table = {
            'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
            'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
            'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
            'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
            'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
            'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
            'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
            'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
            'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
            'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
            'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
            'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
            'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
            'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
            'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
            'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
        }
        self.synonymous_table = {
            '*': ['TAA', 'TAG', 'TGA'],
            'A': ['GCA', 'GCC', 'GCG', 'GCT'],
            'C': ['TGC', 'TGT'],
            'D': ['GAC', 'GAT'],
            'E': ['GAA', 'GAG'],
            'F': ['TTC', 'TTT'],
            'G': ['GGA', 'GGC', 'GGG', 'GGT'],
            'H': ['CAC', 'CAT'],
            'I': ['ATA', 'ATC', 'ATT'],
            'K': ['AAA', 'AAG'],
            'L': ['CTA', 'CTC', 'CTG', 'CTT', 'TTA', 'TTG'],
            'M': ['ATG'],
            'N': ['AAC', 'AAT'],
            'P': ['CCA', 'CCC', 'CCG', 'CCT'],
            'Q': ['CAA', 'CAG'],
            'R': ['AGA', 'AGG', 'CGA', 'CGC', 'CGG', 'CGT'],
            'S': ['AGC', 'AGT', 'TCA', 'TCC', 'TCG', 'TCT'],
            'T': ['ACA', 'ACC', 'ACG', 'ACT'],
            'V': ['GTA', 'GTC', 'GTG', 'GTT'],
            'W': ['TGG'],
            'Y': ['TAC', 'TAT']
        }
        self.degeneracy = ['low', 'med', 'high', '*']
        self.degeneracy_table = {
            'ATA':'med', 'ATC':'med', 'ATT':'med', 'ATG':'low',
            'ACA':'med', 'ACC':'med', 'ACG':'med', 'ACT':'med',
            'AAC':'low', 'AAT':'low', 'AAA':'low', 'AAG':'low',
            'AGC':'high', 'AGT':'high', 'AGA':'high', 'AGG':'high',
            'CTA':'high', 'CTC':'high', 'CTG':'high', 'CTT':'high',
            'CCA':'med', 'CCC':'med', 'CCG':'med', 'CCT':'med',
            'CAC':'low', 'CAT':'low', 'CAA':'low', 'CAG':'low',
            'CGA':'high', 'CGC':'high', 'CGG':'high', 'CGT':'high',
            'GTA':'med', 'GTC':'med', 'GTG':'med', 'GTT':'med',
            'GCA':'med', 'GCC':'med', 'GCG':'med', 'GCT':'med',
            'GAC':'low', 'GAT':'low', 'GAA':'low', 'GAG':'low',
            'GGA':'med', 'GGC':'med', 'GGG':'med', 'GGT':'med',
            'TCA':'high', 'TCC':'high', 'TCG':'high', 'TCT':'high',
            'TTC':'low', 'TTT':'low', 'TTA':'high', 'TTG':'high',
            'TAC':'low', 'TAT':'low', 'TAA':'*', 'TAG':'*',
            'TGC':'low', 'TGT':'low', 'TGA':'*', 'TGG':'low',
        }   

        self.selected_features = [
            ("nt", "C"),
            ("nt", "T"),
            ("nt", "G"),
            ("dnt", "TA"),
            ("dnt", "TG"),
            ("dnt", "TT"),
            ("dnt", "GA"),
            ("dnt", "CT"),
            ("codons", "ATA"),
            ("codons", "AGA"),
            ("codons", "CTA"),
            ("codons", "CGC"),
            ("codons", "TCT"),
            ("codons", "TTA"),
            ("codons", "TAA"),
            ("codons", "TAG"),
            ("codons", "TGA"),
            ("codons", "GAA"),
            ("codons", "GAG"),
            ("codons", "TGG"),
            ("amino", "E"),
            ("amino", "M"),
            ("amino", "D"),
            ("amino", "P"),
            ("amino", "W"),
            ("amino", "Y"),
            ("amino", "I"),
            ("amino", "L"),
            ("synonymous", "TAA"),
            ("synonymous", "CAC"),
            ("synonymous", "ATA"),
            ("synonymous", "CTA"),
            ("synonymous", "AGA"),
            ("synonymous", "TAG"),
            ("synonymous", "CAT"),
            ("synonymous", "TGA"),
            ("synonymous", "GGG"),
            ("synonymous", "GTG"),
            ("synonymous", "CGC"),
            ("synonymous", "TTA"),
            ("synonymous", "TCT"),
            ("synonymous", "GCA"),
            ("synonymous", "GGA"),
            ("synonymous", "AAA"),
            ("synonymous", "CCA"),
            ("synonymous", "GCC"),
            ("synonymous", "ATC"),
            ("synonymous", "AAG"),
            ("synonymous", "CTG"),
            ("synonymous", "CGG"),
            ("synonymous", "TTG"),
            ("synonymous", "TCA"),
            ("synonymous", "GCT"),
            ("synonymous", "GTT"),
            ("degeneracy", "low"),
            ("degeneracy", "med"),
        ]

    def translate(self, trxp_seq, frame):
        amino_acids = list()
        for idx in list(range(frame, len(trxp_seq), 3)):
            codon = trxp_seq[idx:idx+3]
            if len(codon) == 3 :
                amino_acids.append(self.codon_table[codon])
        return ''.join(amino_acids)

    def get_nt_cnt(self, dna):
        return {n:dna.count(n) for n in self.nucleotides}

    def get_nt_frq(self, dna):
        nt_counts = self.get_nt_cnt(dna)
        return {n:nt_counts[n]/sum(list(nt_counts.values())) for n in self.nucleotides}

    def get_nt_cnts(self, trx_seqs):
        nt_cnt = {'A': [], 'C': [], 'T': [], 'G': []}
        for sequence in trx_seqs:
            nt_counts = self.get_nt_cnt(sequence)
            for nt, value in nt_counts.items():
                nt_cnt[nt].append(value)
        return nt_cnt

    def get_nt_frqs(self, trx_seqs):
        nt_frq = {'A': [], 'C': [], 'T': [], 'G': []}
        for sequence in trx_seqs:
            nt_content = self.get_nt_frq(sequence)
            for nt, value in nt_content.items():
                nt_frq[nt].append(value)
        return nt_frq

    def get_dnt_cnt(self, dna):
        return {n:dna.count(n) for n in self.dinucleotides}

    def get_dnt_frq(self, dna):
        dnt_counts = self.get_dnt_cnt(dna)
        return {n:dnt_counts[n]/sum(list(dnt_counts.values())) for n in self.dinucleotides}

    def get_dnt_cnts(self, trx_seqs):
        dnt_cnt = {'AT': [], 'AC': [], 'AG': [], 'AA': [], 'TA': [], 'TC': [], 'TG': [], 'TT': [], 'GT': [], 'GA': [], 'GC': [], 'GG': [], 'CA': [], 'CT': [], 'CG': [], 'CC': []}
        for sequence in trx_seqs:
            dnt_counts = self.get_dnt_cnt(sequence)
            for dnt, value in dnt_counts.items():
                dnt_cnt[dnt].append(value)
        return dnt_cnt

    def get_dnt_frqs(self, trx_seqs):
        dnt_frq = {'AT': [], 'AC': [], 'AG': [], 'AA': [], 'TA': [], 'TC': [], 'TG': [], 'TT': [], 'GT': [], 'GA': [], 'GC': [], 'GG': [], 'CA': [], 'CT': [], 'CG': [], 'CC': []}
        for sequence in trx_seqs:
            dnt_content = self.get_dnt_frq(sequence)
            for dnt, value in dnt_content.items():
                dnt_frq[dnt].append(value)
        return dnt_frq

    def get_codons_cnt(self, dna, frame):
        codons_list = list()
        for idx in list(range(frame, len(dna), 3)):
            codon = dna[idx:idx+3]
            if len(codon) == 3:
                codons_list.append(codon)
        codons_counts = dict(zip(self.codons, [0]*len(self.codons)))
        codons_counts.update(dict(Counter(codons_list)))
        return codons_counts

    def get_codons_frq(self, dna, frame):      
        codons_counts = self.get_codons_cnt(dna, frame)
        codons_contents = {n:codons_counts[n]/sum(list(codons_counts.values())) for n in codons_counts.keys()}
        return codons_contents

    def get_codons_cnts(self, trx_seqs):
        codons_cnt = {'ATA': [], 'ATC': [], 'ATT': [], 'ATG': [], 'ACA': [], 'ACC': [], 'ACG': [], 'ACT': [], 'AAC': [], 'AAT': [], 'AAA': [], 'AAG': [], 'AGC': [], 'AGT': [], 'AGA': [], 'AGG': [], 'CTA': [], 'CTC': [], 'CTG': [], 'CTT': [], 'CCA': [], 'CCC': [], 'CCG': [], 'CCT': [], 'CAC': [], 'CAT': [], 'CAA': [], 'CAG': [], 'CGA': [], 'CGC': [], 'CGG': [], 'CGT': [], 'GTA': [], 'GTC': [], 'GTG': [], 'GTT': [], 'GCA': [], 'GCC': [], 'GCG': [], 'GCT': [], 'GAC': [], 'GAT': [], 'GAA': [], 'GAG': [], 'GGA': [], 'GGC': [], 'GGG': [], 'GGT': [], 'TCA': [], 'TCC': [], 'TCG': [], 'TCT': [], 'TTC': [], 'TTT': [], 'TTA': [], 'TTG': [], 'TAC': [], 'TAT': [], 'TAA': [], 'TAG': [], 'TGC': [], 'TGT': [], 'TGA': [], 'TGG': []}
        for sequence in trx_seqs:
            cnts = []
            for frame in [0, 1, 2]:
                cnts.append(self.get_codons_cnt(sequence, frame))
            for codon in self.codons: 
                cnt = np.sum([cnt[codon] for cnt in cnts])
                codons_cnt[codon].append(cnt)
        return codons_cnt

    def get_codons_frqs(self, trx_seqs):
        codons_frq = {'ATA': [], 'ATC': [], 'ATT': [], 'ATG': [], 'ACA': [], 'ACC': [], 'ACG': [], 'ACT': [], 'AAC': [], 'AAT': [], 'AAA': [], 'AAG': [], 'AGC': [], 'AGT': [], 'AGA': [], 'AGG': [], 'CTA': [], 'CTC': [], 'CTG': [], 'CTT': [], 'CCA': [], 'CCC': [], 'CCG': [], 'CCT': [], 'CAC': [], 'CAT': [], 'CAA': [], 'CAG': [], 'CGA': [], 'CGC': [], 'CGG': [], 'CGT': [], 'GTA': [], 'GTC': [], 'GTG': [], 'GTT': [], 'GCA': [], 'GCC': [], 'GCG': [], 'GCT': [], 'GAC': [], 'GAT': [], 'GAA': [], 'GAG': [], 'GGA': [], 'GGC': [], 'GGG': [], 'GGT': [], 'TCA': [], 'TCC': [], 'TCG': [], 'TCT': [], 'TTC': [], 'TTT': [], 'TTA': [], 'TTG': [], 'TAC': [], 'TAT': [], 'TAA': [], 'TAG': [], 'TGC': [], 'TGT': [], 'TGA': [], 'TGG': []}
        for sequence in trx_seqs:
            frqs = []
            for frame in [0, 1, 2]:
                frqs.append(self.get_codons_frq(sequence, frame))
            for codon in self.codons: 
                frq = np.mean([frq[codon] for frq in frqs])
                codons_frq[codon].append(frq)
        return codons_frq

    def get_amino_cnt(self, amino):
        return {n:amino.count(n) for n in self.amino_acids}
    
    def get_amino_frq(self, amino):
        amino_counts = self.get_amino_cnt(amino)
        return {n:amino_counts[n]/(sum(amino_counts.values())) for n in self.amino_acids}

    def get_amino_cnts(self, trx_seqs):
        amino_cnt = {'A': [], 'R': [], 'N': [], 'D': [], 'C': [], 'Q': [], 'E': [], 'G': [], 'H': [], 'I': [], 'L': [], 'K': [], 'M': [], 'F': [], 'P': [], 'S': [], 'T': [], 'W': [], 'Y': [], 'V': [], '*': []}
        for sequence in trx_seqs:
            cnts = []
            for frame in [0, 1, 2]:
                cnts.append(self.get_amino_cnt(self.translate(sequence, frame)))
            for amino in self.amino_acids: 
                cnt = np.sum([cnt[amino] for cnt in cnts])
                amino_cnt[amino].append(cnt)
        return amino_cnt

    def get_amino_frqs(self, trx_seqs):
        amino_frq = {'A': [], 'R': [], 'N': [], 'D': [], 'C': [], 'Q': [], 'E': [], 'G': [], 'H': [], 'I': [], 'L': [], 'K': [], 'M': [], 'F': [], 'P': [], 'S': [], 'T': [], 'W': [], 'Y': [], 'V': [], '*': []}
        for sequence in trx_seqs:
            frqs = []
            for frame in [0, 1, 2]:
                frqs.append(self.get_amino_frq(self.translate(sequence, frame)))
            for amino in self.amino_acids: 
                frq = np.mean([frq[amino] for frq in frqs])
                amino_frq[amino].append(frq)
        return amino_frq

    def get_synonymous_cnt(self, dna, amino, frame):
        codons_list = []
        for idx in list(range(frame, len(dna), 3)):
            codon = dna[idx:idx+3]
            if len(codon) == 3:
                codons_list.append(codon)
        aa_codon_count = dict()
        for codon in self.synonymous_table[amino]:
            aa_codon_count[codon] = codons_list.count(codon)
        return aa_codon_count

    def get_synonymous_frq(self, dna, amino, frame):
        aa_codon_count = self.get_synonymous_cnt(dna, amino, frame)
        total = sum(aa_codon_count.values())
        aa_codon_frq = dict(zip(self.synonymous_table[amino], [0]*len(self.synonymous_table[amino])))
        aa_codon_frq.update({codon:n/total for codon, n in aa_codon_count.items() if total != 0})
        return aa_codon_frq

    def get_synonymous_cnts(self, trx_seqs):
        codons_cnt = {'ATA': [], 'ATC': [], 'ATT': [], 'ATG': [], 'ACA': [], 'ACC': [], 'ACG': [], 'ACT': [], 'AAC': [], 'AAT': [], 'AAA': [], 'AAG': [], 'AGC': [], 'AGT': [], 'AGA': [], 'AGG': [], 'CTA': [], 'CTC': [], 'CTG': [], 'CTT': [], 'CCA': [], 'CCC': [], 'CCG': [], 'CCT': [], 'CAC': [], 'CAT': [], 'CAA': [], 'CAG': [], 'CGA': [], 'CGC': [], 'CGG': [], 'CGT': [], 'GTA': [], 'GTC': [], 'GTG': [], 'GTT': [], 'GCA': [], 'GCC': [], 'GCG': [], 'GCT': [], 'GAC': [], 'GAT': [], 'GAA': [], 'GAG': [], 'GGA': [], 'GGC': [], 'GGG': [], 'GGT': [], 'TCA': [], 'TCC': [], 'TCG': [], 'TCT': [], 'TTC': [], 'TTT': [], 'TTA': [], 'TTG': [], 'TAC': [], 'TAT': [], 'TAA': [], 'TAG': [], 'TGC': [], 'TGT': [], 'TGA': [], 'TGG': []}
        for sequence in trx_seqs:
            for amino, codons, in self.synonymous_table.items():
                cnts = []
                for frame in [0, 1, 2]:
                    cnts.append(self.get_synonymous_cnt(sequence, amino, frame))
                for codon in codons: 
                    cnt = np.sum([cnt[codon] for cnt in cnts])
                    codons_cnt[codon].append(cnt)
        return codons_cnt
    
    def get_synonymous_frqs(self, trx_seqs):
        codons_frq = {'ATA': [], 'ATC': [], 'ATT': [], 'ATG': [], 'ACA': [], 'ACC': [], 'ACG': [], 'ACT': [], 'AAC': [], 'AAT': [], 'AAA': [], 'AAG': [], 'AGC': [], 'AGT': [], 'AGA': [], 'AGG': [], 'CTA': [], 'CTC': [], 'CTG': [], 'CTT': [], 'CCA': [], 'CCC': [], 'CCG': [], 'CCT': [], 'CAC': [], 'CAT': [], 'CAA': [], 'CAG': [], 'CGA': [], 'CGC': [], 'CGG': [], 'CGT': [], 'GTA': [], 'GTC': [], 'GTG': [], 'GTT': [], 'GCA': [], 'GCC': [], 'GCG': [], 'GCT': [], 'GAC': [], 'GAT': [], 'GAA': [], 'GAG': [], 'GGA': [], 'GGC': [], 'GGG': [], 'GGT': [], 'TCA': [], 'TCC': [], 'TCG': [], 'TCT': [], 'TTC': [], 'TTT': [], 'TTA': [], 'TTG': [], 'TAC': [], 'TAT': [], 'TAA': [], 'TAG': [], 'TGC': [], 'TGT': [], 'TGA': [], 'TGG': []}
        for sequence in trx_seqs:
            for amino, codons, in self.synonymous_table.items():
                frqs = []
                for frame in [0, 1, 2]:
                    frqs.append(self.get_synonymous_frq(sequence, amino, frame))
                for codon in codons: 
                    frq = np.mean([frq[codon] for frq in frqs])
                    codons_frq[codon].append(frq)
        return codons_frq

    def get_degeneracy_cnt(self, dna, frame):
        codons_list = list()
        for idx in list(range(frame, len(dna), 3)):
            codon = dna[idx:idx+3]
            if len(codon) == 3:
                codons_list.append(codon)
        degeneracy_list = list()
        for codon in codons_list:
            degeneracy_list.append(self.degeneracy_table[codon])
        codons_counts = dict(zip(self.degeneracy, [0]*len(self.degeneracy)))
        codons_counts.update(dict(Counter(degeneracy_list)))
        return codons_counts

    def get_degeneracy_frq(self, dna, frame):      
        codons_counts = self.get_degeneracy_cnt(dna, frame)
        total = sum(codons_counts.values())
        codons_contents = {codon:n/total for codon, n in codons_counts.items() if total != 0}
        return codons_contents

    def get_degeneracy_cnts(self, trx_seqs):
        degeneracy_cnt = {'low': [], 'med': [], 'high': [], '*': []} 
        for sequence in trx_seqs:
            cnts = []
            for frame in [0, 1, 2]:
                cnts.append(self.get_degeneracy_cnt(sequence, frame))
            for level in self.degeneracy: 
                cnt = np.sum([cnt[level] for cnt in cnts])
                degeneracy_cnt[level].append(cnt)
        return degeneracy_cnt
    
    def get_degeneracy_frqs(self, trx_seqs):
        degeneracy_frq = {'low': [], 'med': [], 'high': [], '*': []} 
        for sequence in trx_seqs:
            frqs = []
            for frame in [0, 1, 2]:
                frqs.append(self.get_degeneracy_frq(sequence, frame))
            for level in self.degeneracy: 
                frq = np.mean([frq[level] for frq in frqs])
                degeneracy_frq[level].append(frq)
        return degeneracy_frq

    def datasets(self, ensembl_pseudogene):
        datasets = {
            'noncoding': [trx['sequence'] for trx in ensembl_pseudogene.values() if trx['coding'] == 0],
            'coding': [trx['sequence'] for trx in ensembl_pseudogene.values() if trx['coding'] == 1],
            'uncertain': [trx['sequence'] for trx in ensembl_pseudogene.values() if trx['coding'] == 'uncertain']
        }
        return datasets

    def get_cnts_dict(self, datasets):
        cnts_dict = {

            "p": {
                "nt": self.get_nt_cnts(datasets['coding']),
                "dnt": self.get_dnt_cnts(datasets['coding']),
                "codons": self.get_codons_cnts(datasets['coding']),
                "amino": self.get_amino_cnts(datasets['coding']),
                "synonymous": self.get_synonymous_cnts(datasets['coding']),
                "degeneracy": self.get_degeneracy_cnts(datasets['coding'])
            },

            "n": {
                "nt": self.get_nt_cnts(datasets['noncoding']),
                "dnt": self.get_dnt_cnts(datasets['noncoding']),
                "codons": self.get_codons_cnts(datasets['noncoding']),
                "amino": self.get_amino_cnts(datasets['noncoding']),
                "synonymous": self.get_synonymous_cnts(datasets['noncoding']),
                "degeneracy": self.get_degeneracy_cnts(datasets['noncoding'])
            }
        }
        return cnts_dict
    
    def get_all_cnts(self, cnts_dict):
        all_cnts = {
                "p": {
                    "nt": dict(zip(self.nucleotides, len(self.nucleotides)*[0])),
                    "dnt": dict(zip(self.dinucleotides, len(self.dinucleotides)*[0])),
                    "codons": dict(zip(self.codons, len(self.codons)*[0])),
                    "amino": dict(zip(self.amino_acids, len(self.amino_acids)*[0])),
                    "synonymous": dict(zip(self.codons, len(self.codons)*[0])),
                    "degeneracy": dict(zip(self.degeneracy, len(self.degeneracy)*[0]))
                },
                "n": {
                    "nt": dict(zip(self.nucleotides, len(self.nucleotides)*[0])),
                    "dnt": dict(zip(self.dinucleotides, len(self.dinucleotides)*[0])),
                    "codons": dict(zip(self.codons, len(self.codons)*[0])),
                    "amino": dict(zip(self.amino_acids, len(self.amino_acids)*[0])),
                    "synonymous": dict(zip(self.codons, len(self.codons)*[0])),
                    "degeneracy": dict(zip(self.degeneracy, len(self.degeneracy)*[0]))
                }
        }
        for dataset in cnts_dict:
            for feature_type in cnts_dict[dataset]:
                for feature in cnts_dict[dataset][feature_type]:
                    all_cnts[dataset][feature_type][feature] += np.sum(cnts_dict[dataset][feature_type][feature])
        return all_cnts

    def get_chi2_datasets(self, all_cnts):
        chi2_datasets = {
                "p": {
                    "nt": dict(zip(self.nucleotides, len(self.nucleotides)*[0])),
                    "dnt": dict(zip(self.dinucleotides, len(self.dinucleotides)*[0])),
                    "codons": dict(zip(self.codons, len(self.codons)*[0])),
                    "amino": dict(zip(self.amino_acids, len(self.amino_acids)*[0])),
                    "synonymous": dict(zip(self.codons, len(self.codons)*[0])),
                    "degeneracy": dict(zip(self.degeneracy, len(self.degeneracy)*[0]))
                },
                "n": {
                    "nt": dict(zip(self.nucleotides, len(self.nucleotides)*[0])),
                    "dnt": dict(zip(self.dinucleotides, len(self.dinucleotides)*[0])),
                    "codons": dict(zip(self.codons, len(self.codons)*[0])),
                    "amino": dict(zip(self.amino_acids, len(self.amino_acids)*[0])),
                    "synonymous": dict(zip(self.codons, len(self.codons)*[0])),
                    "degeneracy": dict(zip(self.degeneracy, len(self.degeneracy)*[0]))
                }
        }
        for dataset in all_cnts:
            for feature_type in all_cnts[dataset]: 
                if feature_type == "synonymous":
                    for feature in all_cnts[dataset][feature_type]:
                        for amino in self.synonymous_table:
                            if feature in self.synonymous_table[amino]:
                                x = all_cnts[dataset][feature_type][feature]
                                remaining = 0 - x
                                for codon in self.synonymous_table[amino]:
                                    remaining += all_cnts[dataset][feature_type][codon]
                        chi2_datasets[dataset][feature_type][feature] = (x, remaining)        
                else:
                    for feature in all_cnts[dataset][feature_type]:
                        x = all_cnts[dataset][feature_type][feature]
                        remaining = np.sum(list(all_cnts[dataset][feature_type].values())) - x
                        chi2_datasets[dataset][feature_type][feature] = (x, remaining)
        return chi2_datasets

    def get_chi2(self, all_cnts):
        chi2_datasets = self.get_chi2_datasets(all_cnts)
        chi2 = {
                "nt": dict(zip(self.nucleotides, len(self.nucleotides)*[0])),
                "dnt": dict(zip(self.dinucleotides, len(self.dinucleotides)*[0])),
                "codons": dict(zip(self.codons, len(self.codons)*[0])),
                "amino": dict(zip(self.amino_acids, len(self.amino_acids)*[0])),
                "synonymous": dict(zip(self.codons, len(self.codons)*[0])),
                "degeneracy": dict(zip(self.degeneracy, len(self.degeneracy)*[0]))
            }
        sets = list(chi2_datasets.values())
        for feature_type in sets[0].keys():
            for feature in sets[0][feature_type]:
                p = [x[feature_type][feature] for x in sets][0]
                n = [x[feature_type][feature] for x in sets][1]
                if 0 not in p and 0 not in n:
                    x = list((scipy.stats.chi2_contingency([[p[0], p[1]], [n[0], n[1]]]))[:2])
                    chi2[feature_type][feature] = x
        chi2.pop('amino', None)
        return chi2

    def get_features(self, chi2, threshold):
        selected_features = list()
        for feature_type in chi2:
            for feature in chi2[feature_type]:
                if type(chi2[feature_type][feature]) == list and chi2[feature_type][feature][0] > threshold:
                    selected_features.append((feature_type, feature))
        return selected_features

    def get_frqs_dict(self, datasets):
        frqs_dict = {
            "c": {
                "nt": self.get_nt_frqs(datasets['coding']),
                "dnt": self.get_dnt_frqs(datasets['coding']),
                "codons": self.get_codons_frqs(datasets['coding']),
                "amino": self.get_amino_frqs(datasets['coding']),
                "synonymous": self.get_synonymous_frqs(datasets['coding']),
                "degeneracy": self.get_degeneracy_frqs(datasets['coding'])
            },

            "n": {
                "nt": self.get_nt_frqs(datasets['noncoding']),
                "dnt": self.get_dnt_frqs(datasets['noncoding']),
                "codons": self.get_codons_frqs(datasets['noncoding']),
                "amino": self.get_amino_frqs(datasets['noncoding']),
                "synonymous": self.get_synonymous_frqs(datasets['noncoding']),
                "degeneracy": self.get_degeneracy_frqs(datasets['noncoding'])
            },

            "u": {
                "nt": self.get_nt_frqs(datasets['uncertain']),
                "dnt": self.get_dnt_frqs(datasets['uncertain']),
                "codons": self.get_codons_frqs(datasets['uncertain']),
                "amino": self.get_amino_frqs(datasets['uncertain']),
                "synonymous": self.get_synonymous_frqs(datasets['uncertain']),
                "degeneracy": self.get_degeneracy_frqs(datasets['uncertain'])
            }
        }
        return frqs_dict

    def get_frqs_table(self, frqs_dict, datasets, qualified_features):
        features_table = []
        for feature_type, feature in qualified_features:
            features_vect = []
            for target in ["c", "n"]:
                features_vect.extend(frqs_dict[target][feature_type][feature])
            features_table.append(features_vect)

        p_features  = []
        for feature in features_table:
            p_features.append(feature[:(len(datasets['coding']))])
        scaler = preprocessing.StandardScaler().fit(np.array(p_features))
        p_scaled = scaler.transform(np.array(p_features))

        n_features  = []
        for feature in features_table:
            n_features.append(feature[(len(datasets['coding'])):])
        scaler = preprocessing.StandardScaler().fit(np.array(n_features))
        n_scaled = scaler.transform(np.array(n_features))

        all_scaled = []
        for x in range(0, len(qualified_features)):
            scaled = p_scaled[x].tolist() + n_scaled[x].tolist()
            all_scaled.append(scaled)
        X = np.asarray(all_scaled).T
        return X
    
    def get_uncertain_frqs_table(self, frqs_dict, qualified_features):
        features_table = []
        for feature_type, feature in qualified_features:
            features_vect = []
            for target in ["u"]:
                features_vect.extend(frqs_dict[target][feature_type][feature])
            features_table.append(features_vect)

        scaler = preprocessing.StandardScaler().fit(np.asarray(features_table))
        u_scaled = scaler.transform(np.asarray(features_table))
        X = u_scaled.T
        return X

    def get_features_df(self, chi2, frqs_dict):
        features = dict()
        for feat_type, feats in chi2.items():
            for feat, values in feats.items():
                if values != 0:
                    features[f'{feat_type}_{feat}'] = {'chi2':values[0],
                                                    'p_value':values[1],
                                                    'c_mean_frq':np.mean(frqs_dict['c'][feat_type][feat]),
                                                    'n_mean_frq':np.mean(frqs_dict['n'][feat_type][feat])}
        features_df = pd.DataFrame({k:v for k,v in sorted(features.items(), key=lambda item:item[1]['chi2'], reverse=True)}).T
        return features_df

    def get_boxplot(self, feature_type, feature, frqs_dict):
        sns.set_theme(style="whitegrid")
        colors = ["#00B050","#F2F2F2"]
        sns.set_palette(sns.color_palette(colors))
        
        c = frqs_dict['c'][feature_type][feature]
        n = frqs_dict['n'][feature_type][feature]
        both_list = c+n

        c_tags = ['coding'] * len(frqs_dict['c'][feature_type][feature])
        n_tags = ['non-coding'] * len(frqs_dict['n'][feature_type][feature])
        both_tags = c_tags+n_tags

        df1 = pd.DataFrame()
        df1['frequency'] = both_list
        df1['label'] = both_tags

        ax = sns.boxplot(x="label", y='frequency', data=df1, width=0.35, palette=colors, fliersize=2)
        ax = sns.swarmplot(x="label", y='frequency', data=df1, color=".65", size=1)
        plt.title(feature_type+" "+feature)