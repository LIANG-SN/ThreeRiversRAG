import os
import sys
import hashlib

def hash_file(filepath, chunk_size=4096):
    """
    Compute the MD5 hash of a file's contents.
    """
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def find_duplicate_txt_files(directory):
    """
    Walk through the directory and its subdirectories,
    compute a hash for each .txt file, and group duplicates.
    Returns a dictionary mapping file hash to a list of file paths.
    """
    hash_dict = {}
    
    # Walk through the directory recursively
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.txt'):
                filepath = os.path.join(root, file)
                try:
                    file_hash = hash_file(filepath)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    continue
                hash_dict.setdefault(file_hash, []).append(filepath)
    
    # Filter out groups that have only one file (i.e., no duplicates)
    duplicate_groups = {h: paths for h, paths in hash_dict.items() if len(paths) > 1}
    return duplicate_groups

def main():
    if len(sys.argv) < 3:
        print("Usage: python remove_duplicates.py <directory> <output_file>")
        sys.exit(1)
    
    target_directory = sys.argv[1]
    output_file = sys.argv[2]
    
    duplicate_groups = find_duplicate_txt_files(target_directory)
    
    if not duplicate_groups:
        print("No duplicate text files found.")
        sys.exit(0)
    
    with open(output_file, "w", encoding="utf-8") as f_out:
        set_idx = 1
        for file_hash, file_list in duplicate_groups.items():
            # In each group, keep the first file and mark the rest as duplicates to be removed.
            kept_file = file_list[0]
            duplicates = file_list[1:]
            
            # Remove duplicate files
            for dup in duplicates:
                try:
                    os.remove(dup)
                except Exception as e:
                    print(f"Error removing {dup}: {e}")
            
            # Print and write the duplicate set information.
            group_info = f"Duplicate set {set_idx}: Kept: {kept_file}; Removed: {duplicates}"
            print(group_info)
            f_out.write(group_info + "\n")
            set_idx += 1
    
    print(f"\nDuplicate sets written to: {output_file}")

if __name__ == "__main__":
    main()
