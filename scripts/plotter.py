import matplotlib.pyplot as plt

class InitPlot:
    def __init__(self, history, epochs_range, elapsed_time):
        self.history = history
        self.epochs_range = epochs_range
        self.elapsed_time = elapsed_time

    def displayPlot(self):
        best_accuracy = max(self.history['accuracy'])
        lowest_loss = min(self.history['loss'])

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel('Épocas')
        ax1.set_ylabel('Erro', color='tab:red')
        ax1.plot(self.epochs_range, self.history['loss'], label='Erro', color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.set_title(f'Erro e Precisão no treinamento\nMenor erro: {lowest_loss:.4f}, Maior precisão: {best_accuracy:.4f}')

        ax1.annotate(f'Menor erro: {lowest_loss:.4f}', 
                    xy=(self.history['loss'].index(lowest_loss) + 1, lowest_loss),
                    xytext=(self.history['loss'].index(lowest_loss) + 1.5, lowest_loss + 0.02),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Precisão', color='tab:blue')
        ax2.plot(self.epochs_range, self.history['accuracy'], label='Precisão', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        ax2.annotate(f'Maior precisão: {best_accuracy:.4f}', 
                    xy=(self.history['accuracy'].index(best_accuracy) + 1, best_accuracy),
                    xytext=(self.history['accuracy'].index(best_accuracy) + 1.5, best_accuracy - 0.02),
                    arrowprops=dict(facecolor='black', shrink=0.05))

        ax2.set_ylim(0, 1)

        timef = self.elapsed_time / 60
        fig.suptitle(f"Tempo de treinamento: {timef:.2f} minutos", fontsize=16)

        plt.tight_layout()
        plt.show()
