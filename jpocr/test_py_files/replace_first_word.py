f1='out_train_ud/num_1.font.exp0 backup.box'
f2='out_train_ud/random_str_data'

with open(f1, 'r', encoding='utf-8') as file1, open(f2, 'r', encoding='utf-8') as file2:
    lines1 = file1.readlines()
    lines2 = file2.readlines()

    with open('output.txt', 'w', encoding='utf-8') as output_file:
        for line1, line2 in zip(lines1, lines2):
            new_line = line2[0] + line1[1:]
            output_file.write(new_line)