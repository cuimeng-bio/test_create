import os
from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.Blast import NCBIXML
import csv

def run_blast(query_fasta, subject_fasta, output_file):
    """
    运行BLASTn比对
    """
    blastn_cline = NcbiblastnCommandline(query=query_fasta, subject=subject_fasta, outfmt=5, out=output_file, culling_limit=1,max_target_seqs=1)
    stdout, stderr = blastn_cline()
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)

def parse_blast_output(output_file, txt_output_file):
    """
    解析BLASTn输出文件，并将结果保存到txt文件中
    """
    def is_within_range(start1, end1, start2, end2):
        # 检查一个范围是否在另一个范围内，同时考虑反向区间
        if start1 > end1:
            start1, end1 = end1, start1
        if start2 > end2:
            start2, end2 = end2, start2
        return start1 >= start2 and end1 <= end2

    with open(output_file) as result_handle:
        blast_records = NCBIXML.parse(result_handle)
        
        # 收集所有结果
        all_alignments = []
        for blast_record in blast_records:
            query_id = blast_record.query
            for alignment in blast_record.alignments:
                for hsp in alignment.hsps:
                    identity = hsp.identities / hsp.align_length * 100  # 百分比identity
                    query_coverage = hsp.align_length / blast_record.query_length * 100  # 百分比query_coverage
                    hit_coverage = hsp.align_length / alignment.length * 100  # 百分比hit_coverage
                    query_start = hsp.query_start
                    query_end = hsp.query_end
                    subject_start = hsp.sbjct_start
                    subject_end = hsp.sbjct_end
                    e_value = hsp.expect
                    bit_score = hsp.bits
                    all_alignments.append({
                        'Query_ID': query_id, 
                        'Subject_ID': alignment.title, 
                        'Identity(%)': f"{identity:.2f}", 
                        'Query_Coverage(%)': f"{query_coverage:.2f}", 
                        'Hit_Coverage(%)': f"{hit_coverage:.2f}", 
                        'Query_Start': query_start, 
                        'Query_End': query_end, 
                        'Subject_Start': subject_start, 
                        'Subject_End': subject_end,
                        'E-value': e_value,
                        'Bit Score': bit_score
                    })
        
        # 过滤结果
        filtered_alignments = []
        for alignment in all_alignments:
            query_id = alignment['Query_ID']
            query_start = alignment['Query_Start']
            query_end = alignment['Query_End']
            in_range = False
            for fa in filtered_alignments:
                if fa['Query_ID'] == query_id and is_within_range(query_start, query_end, fa['Query_Start'], fa['Query_End']):
                    in_range = True
                    break
            if not in_range:
                filtered_alignments.append(alignment)
        
        # 写入过滤后的结果
        with open(txt_output_file, 'w', newline='') as csvfile:
            fieldnames = ['Query_ID', 'Subject_ID', 'Identity(%)', 'Query_Coverage(%)', 'Hit_Coverage(%)', 'Query_Start', 'Query_End', 'Subject_Start', 'Subject_End', 'E-value', 'Bit Score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            for alignment in filtered_alignments:
                writer.writerow(alignment)

if __name__ == '__main__':
    query_fasta = "/newdata/cuim/prophage/UST-3-CFT073/Bacteria_protein_annotation/UST-3-CFT073.fna"
    subject_fasta = "/newdata/cuim/prophage/CFT073.fasta"
    output_file = "/newdata/cuim/prophage/blast_output.xml"
    txt_output_file = "/newdata/cuim/prophage/blast_results2.txt"

    # 运行BLAST并生成输出文件
    run_blast(query_fasta, subject_fasta, output_file)

    # 解析输出文件并提取相似性信息
    parse_blast_output(output_file, txt_output_file)
    print(f"Results saved to {txt_output_file}")
