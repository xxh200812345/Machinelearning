dir='res/out_train_004/'
input_file='num_1.font.exp0.box'
output_file='output_file'

#开始序列号
start_no=17

with open(f"{dir}{input_file}", 'r') as input_file, open(f"{output_file}", 'w') as output_file:
    # 读取每一行并写入输出文件
    for line in input_file:
        line1s=line.split(' ')
        line1s[5]=str(int(line1s[5])+start_no)
        new_line = ' '.join(line1s)+line[-1]

        output_file.write(new_line)
        