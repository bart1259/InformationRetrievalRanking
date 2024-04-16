rm -rf ./data
mkdir ./data
mkdir ./data/walmart-amazon
wget -O ./data/walmart-amazon/gs_train.csv https://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/data/products_\(Walmart-Amazon\)/gs_train.csv
wget -O ./data/walmart-amazon/gs_val.csv https://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/data/products_\(Walmart-Amazon\)/gs_val.csv
wget -O ./data/walmart-amazon/gs_test.csv https://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/data/products_\(Walmart-Amazon\)/gs_test.csv
wget -O ./data/walmart-amazon/records.zip https://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/data/products_\(Walmart-Amazon\)/records.zip
unzip ./data/walmart-amazon/records.zip -d ./data/walmart-amazon
