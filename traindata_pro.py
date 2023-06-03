
#128个16进制512位数中的第一个16进制数 32个16进制数128位的md5
with    open('./data/eval_data_cls.txt', 'w') as f2, \
        open('./data/eval_data.txt', 'r')as f1:
    for i in range(10000):
        line = f1.readline()
        inputtext_md5sum = line.split()
        label_md5sum = inputtext_md5sum[0][0] + ' ' + inputtext_md5sum[1] + '\n'
        f2.write(label_md5sum)
