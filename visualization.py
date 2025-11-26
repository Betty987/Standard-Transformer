
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator, MultipleLocator

def plot_metrics(
    train_losses, 
    val_losses=None, 
    train_perplexities=None,
    val_perplexities=None
):
    """
    Plot training and validation metrics.
    """
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)

    # Loss Plot
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training')
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation')
    
    plt.title('Training/Validation Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    save_path = assets_dir / 'loss_plot.png'
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to: {save_path}")

    #  Perplexity Plot
    if train_perplexities is not None:
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(train_perplexities) + 1)

        plt.plot(epochs, train_perplexities, 'b-', label='Training')
        if val_perplexities:
            plt.plot(epochs, val_perplexities, 'r-', label='Validation')

        plt.title('Training/Validation Perplexity vs. Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.grid(True)

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if max(train_perplexities) > 100:
            ax.yaxis.set_major_locator(MultipleLocator(100))
            
        save_path_ppl = assets_dir / 'perplexity_plot.png'
        plt.savefig(save_path_ppl)
        plt.close()
        print(f"Perplexity plot saved to: {save_path_ppl}")