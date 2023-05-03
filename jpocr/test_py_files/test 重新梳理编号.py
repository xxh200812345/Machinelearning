#filename = "res/out_train_FOR1.3~5/num_1.font.exp0.box"
filename = "res/out_train_FOR1.3~5/num_1.font.exp0.box"
# 打开文件，读取每一行并添加字母a，然后将每一行写回到原文件中
with open(filename, "r+") as file:
    # 读取文件的每一行
    lines = file.readlines()

    groupno = 0
    bk_groupno = 0
    outputs = []
    # 将每一行添加字母a
    for i in range(len(lines)):
        lines[i] = lines[i].strip()
        output = ""
        this_groupno = int(lines[i].split(" ")[-1])
        if i == 0:
            output = (
                lines[i][0 : (len(lines[i]) - len(str(this_groupno)) - 1)]
                + " 0"
            )
        else:
            if this_groupno != bk_groupno:
                groupno+=1
                output = (
                    lines[i][0 : (len(lines[i]) - len(str(this_groupno)) - 1)]
                    + f" {str(groupno)}"
                    + '\n'
                )
                bk_groupno = this_groupno
            else:
                output = (
                    lines[i][0 : (len(lines[i]) - len(str(this_groupno)) - 1)]
                    + f" {str(groupno)}"
                    + '\n'
                )
        print(f"{lines[i]}   {output}")
        outputs.append(output)

    # 将每一行写回到原文件中
    file.seek(0)
    file.writelines(outputs)
    file.truncate()
