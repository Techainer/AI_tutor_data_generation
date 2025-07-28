mkdir models

cd lib_ocr_omnt
dvc pull -r techainer
cp -r models/* ../models/
cd ..

cd lib_layout_db
dvc pull -r techainer
cp -r models/* ../models/
cd ..