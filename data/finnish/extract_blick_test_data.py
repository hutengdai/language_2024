front_unrounded = ['ä', 'ö', 'y']
back = ['a', 'o', 'u']
neutral = ['e', 'i']
vowels = front_unrounded + back + neutral

def is_grammatical(v1, v2):
    # Both vowels are from the same category or any is a neutral vowel
    if (v1 in front_unrounded and v2 in front_unrounded) or \
       (v1 in back and v2 in back) or \
       (v1 in neutral or v2 in neutral):
        return "grammatical"
    # Vowels are from incompatible categories
    return "ungrammatical"


sequences = []

for v1 in vowels:
    for v2 in vowels:
        sequence = f"t {v1} k {v2} z"
        grammaticality = is_grammatical(v1, v2)
        sequences.append(f"{sequence}\t{grammaticality}")

# Writing to blick_test.txt
with open("data/finnish/blick_test.txt", "w", encoding='utf-8') as file:
    for sequence in sequences:
        file.write(sequence + "\n")