#!/bin/sh

if [ $# -eq 0 ]
then
    echo 'Usage: ./clean_notebooks.sh path1.ipynb [path2.ipynb] [...]'
else
    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace $@
fi
