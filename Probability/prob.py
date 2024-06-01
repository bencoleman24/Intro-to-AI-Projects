import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=dict()
    alphabet_upper = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    for letter in alphabet_upper:
        X[letter] = 0
    
    with open (filename,encoding='utf-8') as f:
        text = f.read()

    text_upper = text.upper()
    for letter in text_upper:
        if letter in alphabet_upper:
            X[letter] += 1
    
    return X


letter = "letter.txt"

# X is a dict of letter, count pairs
X = shred(letter)


# Q1
output = "Q1"
alphabet_upper = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
for letter in alphabet_upper:
    output += f"\n{letter} {X[letter]}"
print(output)

vectors = get_parameter_vectors()
e = vectors[0]
s = vectors[1]
pr_e = 0.6
pr_s = 0.4

# A list of the letter counts, index is same as alphabetical order
X_counts = list(X.values())

# Get input prob list
total_letters = sum(X_counts)
X_probs = []
for count in X_counts:
    X_probs.append(count/total_letters)



# Q2 
XA = X_counts[0]
if XA > 0:  
    log_e1 = XA * math.log(e[0])
    log_s1 = XA * math.log(s[0])
else:
    log_e1 = 0 
    log_s1 = 0

log_e1_rounded = round(log_e1,4)
log_s1_rounded = round(log_s1,4)
log_e1_rounded = "{:.4f}".format(log_e1_rounded)
log_s1_rounded = "{:.4f}".format(log_s1_rounded)
print("Q2")
print(log_e1_rounded)
print(log_s1_rounded)

FE = math.log(pr_e)
FS = math.log(pr_s)

# Q3
for i in range(26):
    count = X_counts[i]
    if count != 0:
        log_ei = math.log(e[i])
        log_si = math.log(s[i])
        FE += count * log_ei
        FS += count * log_si
FE_rounded = round(FE, 4)
FS_rounded = round(FS, 4)
FE_rounded = "{:.4f}".format(FE_rounded)
FS_rounded = "{:.4f}".format(FS_rounded)
print("Q3")
print(FE_rounded)
print(FS_rounded)

if FS - FE >= 100:
    prob_e = 0.0
elif FS - FE <= -100:
    prob_e = 1.0
else:
    prob_e = 1 / (1 + math.exp(FS - FE))

prob_e_rounded = round(prob_e, 4)
prob_e_rounded = "{:.4f}".format(prob_e_rounded)
print("Q4")
print(prob_e_rounded)
