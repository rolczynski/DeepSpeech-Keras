wget http://2018.poleval.pl/task3/task3_train.txt.gz -o language_model_training_data.txt.gz
gzip -d language_model_training_data.txt.gz
rm language_model_training_data.txt.gz
cat language_model_training_data.txt | sed -E 's/[^a-zA-Ząćęłóśźż ]//g' | sed 's/\ /@/g' | sed -e 's/\(.\)/\1 /g' > language_model_training_data_preprocessed.txt