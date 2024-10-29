import matplotlib.pyplot as plt

def plot(epochs_range, history, best_accuracy, lowest_loss, elapsed_time):

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Erro', color='tab:red')
    ax1.plot(epochs_range, history.history['loss'], label='Erro', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_title(f'Erro e Precisão no treinamento\nMenor erro: {lowest_loss:.4f}, Maior precisão: {best_accuracy:.4f}')

    ax1.annotate(f'Menor erro: {lowest_loss:.4f}', 
        xy=(history.history['loss'].index(lowest_loss) + 1, lowest_loss),
        xytext=(history.history['loss'].index(lowest_loss) + 1.5, lowest_loss + 0.02),
        arrowprops=dict(facecolor='black', shrink=0.05))
                        
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Precisão', color='tab:blue')
    ax2.plot(epochs_range, history.history['accuracy'], label='Precisão', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    ax2.annotate(f'Maior precisão: {best_accuracy:.4f}', 
        xy=(history.history['accuracy'].index(best_accuracy) + 1, best_accuracy),
        xytext=(history.history['accuracy'].index(best_accuracy) + 1.5, best_accuracy - 0.02),
        arrowprops=dict(facecolor='black', shrink=0.05))

    ax2.set_ylim(0, 1)
                        
    timef = elapsed_time / 60

    fig.suptitle(f"Tempo de treinamento: {timef:.2f} minutos", fontsize=16)