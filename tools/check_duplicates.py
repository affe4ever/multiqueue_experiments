#!/usr/bin/env python3
import csv
from collections import Counter
import sys

"""Can be used to check for duplicate node ids in the output from analyze quality"""

def check_duplicates(csv_file):
    node_ids = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_ids.append(int(row['node_id']))
    
    counts = Counter(node_ids)
    all_counts = list(counts.values())
    duplicates = {node_id: count for node_id, count in counts.items() if count > 1}
    
    # Calculate statistics for ALL nodes
    highest = max(all_counts)
    lowest = min(all_counts)
    average = sum(all_counts) / len(all_counts)
    
    sys.stderr.write(f"Statistics for all nodes:\n")
    sys.stderr.write(f"  Highest count: {highest}\n")
    sys.stderr.write(f"  Lowest count: {lowest}\n")
    sys.stderr.write(f"  Average count: {average:.2f}\n")
    
    if duplicates:
        print(f"Found {len(duplicates)} node(s) with duplicates:")
        for node_id in sorted(duplicates.keys()):
            print(f"  Node {node_id}: {duplicates[node_id]} occurrences")
    else:
        print("No duplicate node_ids found (all nodes appear <= 1 time)")

if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'temp.csv'
    check_duplicates(csv_file)
