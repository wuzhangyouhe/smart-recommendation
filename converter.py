with open('bk-rates.csv') as fin, open('new-bk-rates.csv', 'w') as fout:
    for line in fin:
        fout.write(line.replace('\t', ','))