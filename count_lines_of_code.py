import os

def count_lines_of_code(directory):
    total_lines = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    in_multiline_comment = False
                    for line in f:
                        stripped_line = line.strip()
                        # Check for single-line comments or empty lines
                        if stripped_line.startswith('#') or not stripped_line:
                            continue
                        # Check for multi-line string (docstring) start/end
                        if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                            if in_multiline_comment:
                                in_multiline_comment = False
                            else:
                                in_multiline_comment = True
                            continue
                        if in_multiline_comment:
                            continue
                        total_lines += 1
    return total_lines

# Replace 'rag_pipeline' with the path to your codebase directory
codebase_directory = 'tests'
total_lines = count_lines_of_code(codebase_directory)
print(f"Total lines of code (excluding comments): {total_lines}")