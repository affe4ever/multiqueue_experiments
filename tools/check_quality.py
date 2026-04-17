#!/usr/bin/env python3
import csv
from collections import Counter
import sys

"""Can be used to check for duplicate node ids in the output from analyze quality"""

def check_duplicates(csv_file):
    extra_work_per_node = Counter()
    extra_work_generated_per_node = Counter()
    total_extra_work = 0
    total_extra_work_generated = 0
    total_ignored_work = 0
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(row['node_id'])
            extra_work = int(row['extra_work'])
            extra_work_generated = int(row['extra_work_generated'])
            ignored_node = int(row['ignored_node'])
            extra_work_per_node[node_id] += extra_work
            extra_work_generated_per_node[node_id] += extra_work_generated
            total_extra_work += extra_work
            total_extra_work_generated += extra_work_generated
            total_ignored_work += ignored_node
    
    sys.stderr.write(f"Total extra work: {total_extra_work}\n")
    if extra_work_per_node:
        max_extra_work = max(extra_work_per_node.values())
        avg_extra_work = total_extra_work / len(extra_work_per_node)
        sys.stderr.write(f"Max extra work per node: {max_extra_work}\n")
        sys.stderr.write(f"Average extra work per node: {avg_extra_work:.2f}\n")
    
    sys.stderr.write(f"Total extra work generated: {total_extra_work_generated}\n")
    if extra_work_generated_per_node:
        max_extra_work_generated = max(extra_work_generated_per_node.values())
        avg_extra_work_generated = total_extra_work_generated / len(extra_work_generated_per_node)
        sys.stderr.write(f"Max extra work generated per node: {max_extra_work_generated}\n")
        sys.stderr.write(f"Average extra work generated per node: {avg_extra_work_generated:.2f}\n")
    
    sys.stderr.write(f"Total ignored nodes: {total_ignored_work}\n")

if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'temp.csv'
    check_duplicates(csv_file)
