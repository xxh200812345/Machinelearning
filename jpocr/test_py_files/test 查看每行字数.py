with open("res/allforone copy/random_str_data", "r") as f:
    for i, line in enumerate(f, 1):
        print(f"Line {i}: {line.strip()} ({len(line)-1} words)")