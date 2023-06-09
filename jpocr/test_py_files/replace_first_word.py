f1='res/allforone copy/num_1.font.exp0.box'
f2='res/allforone copy/random_str_data'
shift_num=0

with open(f1, 'r', encoding='utf-8') as file1, open(f2, 'r', encoding='utf-8') as file2:
    lines1 = file1.readlines()
    lines2 = file2.readlines()
    letters = []
    for line in lines2:
        letters += list(line.strip())

    print(f"lines1:{len(lines1)},lines2:{len(letters)}")

    with open('output.txt', 'w', encoding='utf-8') as output_file:
        for line1, line2 in zip(lines1, letters):
            line1s=line1.split(' ')
            line1s[0]=line2[0]
            line1s[5]=str(int(line1s[5])+shift_num)
            new_line = ' '.join(line1s)+line1[-1]
            output_file.write(new_line)