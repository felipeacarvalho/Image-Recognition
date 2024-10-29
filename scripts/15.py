filename = 'nomes.txt'

with open(filename, 'w') as file:
    for i in range(0, 7):
        for _ in range(1000):
            file.write(f'{i}\n')

print(f"feito")