f1_name = "train_all.txt"
f2_name = "train.txt"
f3_name = "val.txt"
f1 = open(f1_name, 'r')
f2 = open(f2_name, 'w')
f3 = open(f3_name, 'w')

train_num = 3000
val_num = 500
i = 0
for line in f1.readlines():
    i = i + 1
    if i > train_num + val_num:
        break
    if i > train_num:
        f3.write(line)
        continue
    f2.write(line)
f2.close()
f3.close()
f1.close()

