import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses, log_step):
    plt.figure(figsize=(12, 6))
    train_steps = list(range(1, len(train_losses) + 1))
    val_steps = [(i + 1) * log_step for i in range(len(val_losses) - 1)]
    val_steps.append(len(train_losses))  

    plt.plot(train_steps, train_losses, label="Training Loss", color='blue')
    plt.plot(val_steps, val_losses, label="Validation Loss", color='red', marker='o', linestyle='--')

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Over Time")
    plt.legend()
    plt.grid(True)

    plt.show()
