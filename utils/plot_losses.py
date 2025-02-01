import matplotlib.pyplot as plt

def live_plot(train_losses, val_losses, log_step):
    plt.ion() 
    fig, ax = plt.subplots(figsize=(12, 6))

    train_steps = list(range(1, len(train_losses) + 1))
    val_steps = [(i + 1) * log_step for i in range(len(val_losses) - 1)]
    val_steps.append(len(train_losses)) 

    ax.clear()
    ax.plot(train_steps, train_losses, label="Training Loss", color='blue', alpha=0.7)
    ax.plot(val_steps, val_losses, label="Validation Loss", color='red', marker='o', linestyle='--')

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Real-Time Training & Validation Loss")
    ax.legend()
    ax.grid(True)

    plt.draw()
    plt.pause(0.1)
