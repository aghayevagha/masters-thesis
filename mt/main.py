import hashlib
import numpy as np
from IBE import IBE


user_identity = "user@example.com"
#I will change only one letter to see what happens
attacker_identity = "use'@example.com"


#define the IBE
b = IBE(5,10003,5)


# generate a random public vector, normally it should come from user's identity string
#user_v = np.random.randint(0,b.q,size = (b.n,))
user_v = b.identity_to_vector(user_identity, b.n,b.q)
#print(user_v)

# generate secret key
user_sec_key = b.generate_secret_key(user_v)
calculated_user_public_key= (b.mpk @ user_sec_key)%b.q



#if you wanna see the corresponding secret key is correct
print(user_v==calculated_user_public_key.T )


#choose the message and encrypt
message = 1
c1, c2, error_term, e_prime = b.encrypt(message, user_v)


# Here I calculate the result of Dual Regev scheme by hand to see if all calculation is correct,
# the randomized errors should not be shared to public.
# the result is : e' - e_vector.T * user_secret_key + m.(q/2)
calculated_alpha = e_prime - error_term.T @ user_sec_key + message*np.floor(b.q/2) 
calculated_alpha = calculated_alpha % b.q


result = b.decrypt((c1,c2),user_sec_key)
print("alphas:",calculated_alpha, result[1])
print("message",result[0])


print("="*50)
print("TRYING ANOTHER USER'S PUBLIC KEY")

#=====================
# trying another public key of attacker
attacker_v = b.identity_to_vector(attacker_identity, b.n,b.q)
attacker_sec_key = b.generate_secret_key(attacker_v)



# Here I calculate the result of Dual Regev scheme by with attacker's secret key
attack_calculated_alpha = e_prime - error_term.T @ attacker_sec_key + message*np.floor(b.q/2) 
attack_calculated_alpha = calculated_alpha % b.q


attack_result = b.decrypt((c1,c2),attacker_sec_key)
print("alphas:",attack_calculated_alpha, attack_result[1])
print("message",attack_result[0])
