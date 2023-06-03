from datetime import datetime
import random
import hashlib

dt = datetime.now()
print(f"start generating train data {dt.year},{dt.month},{dt.day},{dt.hour}:{dt.minute}:{dt.second}\ndatalen = 512,md5len = 128,sha-256len = 256")

rand_seed = dt.month*12 + dt.day + dt.hour*24 + dt.minute*60 + dt.second
random.seed(rand_seed)

def gen_512_md5():
    random_num = '' 
    for i in range(128):
        random_num = random_num+(random.sample('0123456789abcdef',1))[0]
    md5_val = hashlib.md5(random_num.encode('utf8')).hexdigest()
    return random_num,md5_val
#128个16进制512位数 32个16进制数128位的md5

with open('./data/eval_data.txt', 'w') as f:
    for i in range(10000):
        random_num,md5_val = gen_512_md5()
        data = random_num + ' ' + md5_val + '\n'
        f.write(data)
