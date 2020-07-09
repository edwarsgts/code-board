# Bin2Dec project
# Arrays cannot be used


# Input validation
is_valid_input = False

while not is_valid_input:
    binary_input = input(
        "Please enter a binary number(consisting only 0 or 1): ")
    for n in binary_input:
        if not(n == '0' or n == '1'):
            print("digit must be 0 or 1. Please re-enter:")
            break
    else:
        is_valid_input = True

dec_ans = 0
for i, n in enumerate(reversed(binary_input)):
    dec_ans += (int(n)*(2**i))

print(f"Your number in decimal number would be: {dec_ans}")
