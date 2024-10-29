import tkinter as tk
from tkinter import filedialog

class SelecDir:

    def expArq():
        root = tk.Tk()
        root.withdraw()  

        tkDir = filedialog.askdirectory(title="Selecione uma pasta")

        print(f"Diretório selecionado: {tkDir}")
        
        return tkDir

class SelecArq:

    def expArq():
        root = tk.Tk()
        root.withdraw()  

        tkArq = (filedialog.askopenfile(title="Selecione um arquivo")).name

        print(f"Arquivo selecionado: {tkArq}")
        
        return tkArq