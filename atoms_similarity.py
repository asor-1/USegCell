import csv
from collections import defaultdict
from itertools import combinations

def load_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def get_file_prefixes(headers):
    return sorted(set(h.split(',')[0] for h in headers if h.startswith('File')))

def get_atom_key(row, file_prefix):
    try:
        return (
            row[f'{file_prefix},Chain'],
            row[f'{file_prefix},SeqID'],
            row[f'{file_prefix},AA'],
            row[f'{file_prefix},Atom']
        )
    except KeyError:
        return None

def find_matching_atoms(data):
    file_prefixes = get_file_prefixes(data[0].keys())
    atom_locations = defaultdict(lambda: defaultdict(set))

    for row in data:
        for prefix in file_prefixes:
            atom_key = get_atom_key(row, prefix)
            if atom_key and all(atom_key):  # Ensure all parts of the key are present
                atom_locations[prefix][atom_key].add(row[prefix])

    return atom_locations

def compare_files(atom_locations):
    file_prefixes = list(atom_locations.keys())
    comparisons = defaultdict(list)

    for file1, file2 in combinations(file_prefixes, 2):
        common_atoms = set(atom_locations[file1].keys()) & set(atom_locations[file2].keys())
        for atom in common_atoms:
            locations1 = atom_locations[file1][atom]
            locations2 = atom_locations[file2][atom]
            if locations1 & locations2:  # If there's any overlap in locations
                comparisons[(file1, file2)].append((atom, locations1 & locations2))

    return comparisons

def print_results(comparisons):
    for (file1, file2), matches in comparisons.items():
        print(f"\nMatching atoms between {file1} and {file2}:")
        for i, (atom, locations) in enumerate(matches, 1):
            chain, seq_id, aa, atom_type = atom
            locations_str = ", ".join(sorted(locations))
            print(f"{i}. Chain: {chain}, SeqID: {seq_id}, AA: {aa}, Atom: {atom_type}")
            print(f"   Locations: {locations_str}")
        print(f"Total matches: {len(matches)}")

def main():
    filename = "Untitled spreadsheet - Sheet1.csv"
    data = load_csv(filename)
    atom_locations = find_matching_atoms(data)
    comparisons = compare_files(atom_locations)
    print_results(comparisons)

if __name__ == "__main__":
    main()