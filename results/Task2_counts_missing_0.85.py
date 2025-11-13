count = 0
total = 0

with open("temp.txt", "r") as f:
    lines = [line.strip() for line in f if line.strip()]  # remove empty lines

for i in range(0, len(lines) - 1, 2):  # step by 2, and avoid out-of-bounds
    if "obj_pos" in lines[i] and "cmd_pos" in lines[i + 1]:
        obj_line = lines[i]
        cmd_line = lines[i + 1]

        try:
            obj_x = float(obj_line.split('tensor([[ ')[1].split(',')[0])
            cmd_x = float(cmd_line.split('tensor([[ ')[1].split(',')[0])
            if obj_x <= cmd_x:
                count += 1
            total += 1
        except Exception as e:
            print(f"Failed to parse lines:\n{obj_line}\n{cmd_line}\nError: {e}")

print(f"Valid comparisons: {total}")
print(f"Number of times obj_x ≤ cmd_x: {count}")
