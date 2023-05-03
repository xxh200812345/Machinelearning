#!/bin/bash

#在生产环境中运行

# Set the paths
JTESSBOXEDITOR=./jTessBoxEditor.jar
TRAIN_IMAGE_NAME=num_1
TRAIN_IMAGE=${TRAIN_IMAGE_NAME}.font.exp0.tif
TRAIN_LABEL=${TRAIN_IMAGE_NAME}.font.exp0
TRAIN_BOX=${TRAIN_IMAGE_NAME}.font.exp0.box
UNINCHAR_SET=${TRAIN_IMAGE_NAME}.unicharset
TR=${TRAIN_IMAGE_NAME}.font.exp0.tr
TESSDATA=./tessdata
LANG=mylang
FONT=myfont

# Create the box file
echo "font 0 0 0 0 0">font_properties
tesseract  $TRAIN_IMAGE $TRAIN_LABEL nobatch box.train
unicharset_extractor $TRAIN_BOX
mftraining -F font_properties -U unicharset -O $UNINCHAR_SET $TR
cntraining $TR

# Create the traineddata file
mv inttemp ${TRAIN_IMAGE_NAME}.inttemp
mv pffmtable ${TRAIN_IMAGE_NAME}.pffmtable
mv normproto ${TRAIN_IMAGE_NAME}.normproto
mv shapetable ${TRAIN_IMAGE_NAME}.shapetable

combine_tessdata ${TRAIN_IMAGE_NAME}.