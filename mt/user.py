import IBE
import numpy as np
class User:
    def __init__(self, identity):
        self.identity = identity
        self.sec_key =  None

    def assign_key( self, x):
        self.sec_key = x

    def recieve_message (self, ciphertext):
        message = IBE.decrypt(ciphertext,self.sec_key)
    
