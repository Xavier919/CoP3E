import csv
from Bio import SeqIO
import pyfaidx
import pickle
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm_notebook as tqdm


class BuildDataset:
    def __init__(self, OP_tsv, Ens_trx, trx_fasta, altprot_pep_any_unique):
        
        self.biotype_grouping = {
            'protein_coding': 'protein_coding',
            'processed_transcript': 'processed_transcript',
            'miRNA': 'processed_transcript',
            'misc_RNA': 'processed_transcript',
            'unprocessed_pseudogene': 'pseudogene',
            'antisense': 'processed_transcript',
            'retained_intron': 'processed_transcript',
            'processed_pseudogene': 'pseudogene',
            'rRNA_pseudogene': 'pseudogene',
            'sense_intronic': 'processed_transcript',
            'lincRNA': 'processed_transcript',
            'snoRNA': 'processed_transcript',
            'snRNA': 'processed_transcript',
            'transcribed_unprocessed_pseudogene': 'pseudogene',
            'translated_processed_pseudogene': 'pseudogene',
            'nonsense_mediated_decay': 'nmd',
            'polymorphic_pseudogene': 'pseudogene',
            'transcribed_processed_pseudogene': 'pseudogene',
            'TEC': 'others',
            'sense_overlapping': 'processed_transcript',
            'TR_J_gene': 'others',
            'IG_J_gene': 'others',
            'IG_V_pseudogene': 'pseudogene',
            'TR_V_gene': 'others',
            'rRNA': 'processed_transcript',
            'TR_C_gene': 'others',
            'scaRNA': 'processed_transcript',
            'IG_V_gene': 'others',
            'pseudogene': 'pseudogene',
            'bidirectional_promoter_lncRNA': 'processed_transcript',
            'TR_V_pseudogene': 'pseudogene',
            'Mt_tRNA': 'processed_transcript',
            'unitary_pseudogene': 'pseudogene',
            'IG_C_gene': 'others',
            'IG_pseudogene': 'pseudogene',
            'transcribed_unitary_pseudogene': 'pseudogene',
            'IG_C_pseudogene': 'pseudogene',
            'IG_D_gene': 'others',
            'non_coding': 'processed_transcript',
            'ribozyme': 'processed_transcript',
            '3prime_overlapping_ncRNA': 'processed_transcript',
            'TR_J_pseudogene': 'pseudogene',
            'IG_J_pseudogene': 'pseudogene',
            'non_stop_decay': 'nmd',
            'sRNA': 'processed_transcript',
            'TR_D_gene': 'others',
            'scRNA': 'processed_transcript',
            'vaultRNA': 'processed_transcript',
            'Mt_rRNA': 'processed_transcript',
            'macro_lncRNA': 'processed_transcript',
            'lincrna': 'processed_transcript',
            }
        
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

        self.OP_tsv = OP_tsv
        self.Ens_trx = Ens_trx
        self.ensembl95_trxps = pyfaidx.Fasta(trx_fasta)
        self.altprot_pep_any_unique = pickle.load(open(altprot_pep_any_unique, 'rb'))

        self.OP_prot_MS, self.OP_trx_altprot = self.get_altprot_info()

    def find_orfs(self, seq, thresh):
        start_codons, stop_codons = ['ATG'], ['TGA', 'TAA', 'TAG']
        frames = [0,1,2]
        orfs = []
        for frame in frames:
            starts, stops = [], []
            for idx in list(range(frame, len(seq), 3)):
                codon = seq[idx:idx+3]
                if codon in start_codons: 
                    starts.append(idx)
                elif codon in stop_codons:
                    stops.append(idx+3)
            stops = stops[::-1]
            for idx, stop in enumerate(stops):
                for start in starts:
                    if stop - start < thresh or any(i > start for i in stops[idx+1:]):
                        continue
                    else:
                        orfs.append((start, stop))
                        break
        return orfs

    def get_op_trx(self):
        op_trx = set()
        with open(self.OP_tsv, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for n, row in enumerate(reader):
                if n==0: continue
                if n==1:
                    cols = row
                    continue
                line = dict(zip(cols, row)) 
                if "ENST" in line['transcript accession'] :
                    line['transcript accession'] = line['transcript accession'].split('.')[0]
                    op_trx.add(line['transcript accession'])
        return op_trx

    def get_max_evidence_level(self, trxp_id):
        ev_levs = list()
        for alt_accession in self.OP_trx_altprot[trxp_id]:
            MS = self.OP_prot_MS[alt_accession]['MS score']
            TE = self.OP_prot_MS[alt_accession]['TE score']
            ev_lev = 0
            if MS > 0 or TE > 0:
                ev_lev = 1
            if MS >= 2 or TE >= 2:
                ev_lev = 2   
            if alt_accession in self.altprot_pep_any_unique and not self.altprot_pep_any_unique[alt_accession]:
                ev_lev = 0
            ev_levs.append(ev_lev)
        return max(ev_levs)

    def get_altprot_info(self):
        OP_prot_MS = dict()
        OP_trx_altprot = dict()
        with open(self.OP_tsv, 'r') as f:
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
                if not any(x in line["protein accession numbers"] for x in ["IP_", "II_", "ENSP"]):
                    continue 
                trx_stable_ID = line["transcript accession"].split(".")[0]
                if trx_stable_ID not in OP_trx_altprot:
                    OP_trx_altprot[trx_stable_ID] = [line["protein accession numbers"]]
                else:
                    OP_trx_altprot[trx_stable_ID].append(line["protein accession numbers"])
                    
                OP_prot_MS[line["protein accession numbers"]] = {
                    'MS score': int(line["MS score"]), 
                    'TE score': int(line["TE score"]),
                    }
            return OP_prot_MS, OP_trx_altprot

    def ensembl_trx(self):
        op_trx = self.get_op_trx()
        ensembl_trx = dict()
        with open(self.Ens_trx, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for n, row in tqdm(enumerate(reader)):
                if n==0:
                    cols = row
                    continue
                line = dict(zip(cols, row))
                trx = line['Transcript stable ID']
                sequence = str(self.ensembl95_trxps["|".join([line['Gene stable ID'], line['Transcript stable ID']])])
                if 'N' in sequence:
                    continue
                has_pred_orf, evidence = None, None
                n_orfs = len(self.find_orfs(sequence, 90))
                orf_accessions = []
                if n_orfs > 0 or line["Transcript type"] == 'protein_coding':
                    has_pred_orf = 1
                    if trx in op_trx and trx in self.OP_trx_altprot:
                        evidence = self.get_max_evidence_level(trx)
                        orf_accessions = self.OP_trx_altprot[trx]
                    else: 
                        evidence = "?"
                else:
                    has_pred_orf, evidence = 0, 0
                ensembl_trx[trx] = {'gene_id':line["Gene stable ID"],
                                    'has_pred_orf': has_pred_orf,
                                    'gene_name': line["Gene name"],
                                    'biotype': self.biotype_grouping[line["Transcript type"]],
                                    'biotype_ungroup': line["Transcript type"],
                                    'orf_accessions': orf_accessions,
                                    'evidence': evidence,
                                    'sequence': sequence}
        return ensembl_trx

    def ensembl_pseudogene(self, ensembl_trx):
        ensembl_pseudogene = dict()
        for trx, attrs in ensembl_trx.items():
            ev = attrs['evidence']
            has_pred_orf = attrs['has_pred_orf']
            coding = ''
            if attrs['biotype'] == 'pseudogene':
                if has_pred_orf == 1 and ev == 2:
                    coding = 1
                elif has_pred_orf == 0:
                    coding = 0
                else:
                    coding = 'uncertain'
                ensembl_pseudogene[trx] = {'coding': coding, 'sequence': attrs['sequence']}
        return ensembl_pseudogene

    def pc_datasets(self, ensembl_pseudogene, figsize=8):
        counts = list(Counter([x['coding'] for x in ensembl_pseudogene.values()]).items())
        counts.sort(key=lambda x:x[1]*-1)
        label_counts = list(zip(["Low or no evidence", "No ORF", "High evidence"], [x[1] for x in counts]))
        labels = [x[0] for x in label_counts]
        counts = [x[1] for x in label_counts]
        mycolors = ["#ffeeaaff", "#F2F2F2", "#00B050"]
        fig1, ax1 = plt.subplots(figsize=(figsize,figsize))
        ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors = mycolors)
        ax1.axis('equal')  
        plt.show()