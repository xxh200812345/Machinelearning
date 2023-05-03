#!/bin/bash


# Define file path
traineddata_path=num_1.traineddata

# Append timestamp to file name
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
new_traineddata_path=traineddata/${traineddata_path%.*}_${timestamp}.${traineddata_path##*.}
echo "Copying file to: $new_traineddata_path"
# Copy file to new path with timestamp in name
cp $traineddata_path $new_traineddata_path

#利用.tif和.box文件生成.lstmf文件用于lstm训练
tesseract num_1.font.exp0.tif num_1.font.exp0.box --psm 6 lstm.train

#从已有的.traineddata中提取.lstm文件
combine_tessdata -e num_1.traineddata num_1.lstm

#创建num.training_files文件，里边的内容为.lstmf文件的路径地址
lstmf_filepath=$(readlink -f *.lstmf)
echo $lstmf_filepath > num.training_files

#进行训练

# Define folder path
output_dir=$(pwd)/output
# Create folder if it does not exist
rm -r $output_dir
mkdir -p $output_dir

#--modeloutput 模型训练输出的路径
#--continue_from 训练从哪里继续，这里指定从上面提取的 num_1.lstm文件，
#--train_listfile 指定上一步创建的文件的路径
#--traineddata 指定.traineddata文件的路径
#--debug_interval 当值为-1时，训练结束，会显示训练的一些结果参数
#--max_iterations 指明训练遍历次数
lstmtraining --model_output="output/output" --continue_from="num_1.lstm" --train_listfile="num.training_files" --traineddata="num_1.traineddata" --debug_interval -1 --max_iterations 400

#将checkpoint文件和.traineddata文件合并成新的.traineddata文件
lstmtraining --stop_training --continue_from="output/output_checkpoint" --traineddata="num_1.traineddata" --model_output="num_1.traineddata"
