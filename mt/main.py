import hashlib
import numpy as np
from IBE import IBE


b = IBE(5,103,5)
# generate a random public vector, normally it should come from user's identity string
user_v = np.random.randint(0,b.q,size = (b.n,))
#print(user_v)

# generate secret key
user_sec_key = b.generate_secret_key(user_v)
check= (b.mpk @ user_sec_key)%b.q
#if you wanna see the corresponding secret key is correct
#print(user_v, check )

message = 0
c = b.encrypt(message, user_v)

print("c",c)
result_m = b.decrypt(c,user_sec_key)

print(result_m)