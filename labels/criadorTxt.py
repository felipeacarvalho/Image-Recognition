filename = 'labels.txt'

qtd = int(input("Quantas imagens há na base? "))
divisoes = int(input("Quantas divisões iguais há entre as imagens? "))
try:
    rg = int(qtd/divisoes)

    with open(filename, 'w') as file:
        for i in range(0, divisoes):
            for _ in range(rg):
                file.write(f'{i}\n')

    print("Arquivo escrito.")

except Exception as e:
    print("A razão de imagens e separações deve ser um número inteiro.")
    print(f'\n {e}')