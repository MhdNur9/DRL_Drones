print("......Starting ......")

import itertools
import pikepdf

pdf_file = "/home/mirpalab-sim/Documents/1234.pdf"

# -----------------------
# 1) Digits-only (0–9)
# -----------------------
digits = "0123456789"
min_length = 1
max_length = 5

print("Phase 1: Trying digits only (0–9)")
for length in range(min_length, max_length + 1):
    print(f"  -> Length {length}")
    for pwd_tuple in itertools.product(digits, repeat=length):
        password = "".join(pwd_tuple)
        try:
            with pikepdf.open(pdf_file, password=password):
                print("\n====================================")
                print("PASSWORD FOUND (digits only):", password)
                print("====================================")
                raise SystemExit
        except pikepdf.PasswordError:
            continue

print("Phase 1 done. No match with digits only.")

# -----------------------------------------
# 2) Arabic letters + digits (very large!)
# -----------------------------------------

# Basic Arabic alphabet (you can extend this if needed)
arabic_letters = "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"

charset = digits + arabic_letters
print(f"\nPhase 2: Trying Arabic letters + digits")
print(f"Charset size: {len(charset)} characters")

for length in range(min_length, max_length + 1):
    print(f"  -> Length {length}")
    for pwd_tuple in itertools.product(charset, repeat=length):
        password = "".join(pwd_tuple)
        try:
            with pikepdf.open(pdf_file, password=password):
                print("\n====================================")
                print("PASSWORD FOUND (Arabic+digits):", password)
                print("====================================")
                raise SystemExit
        except pikepdf.PasswordError:
            continue

print("Password not found in given search space :(")
print("......Done......")
