import matplotlib.pyplot as plt
import re
import os
import argparse

def parse_log(log_path):
    iterations = []
    train_losses = []
    val_iterations = []
    val_losses = []

    # Regex for standard MLX reporting lines
    line_pattern = re.compile(r"Iter (\d+):")
    train_pattern = re.compile(r"Train loss ([\d\.]+)")
    val_pattern = re.compile(r"Val loss ([\d\.]+)")

    if not os.path.exists(log_path):
        print(f"[ERROR] Log file not found at: {log_path}")
        return None, None, None, None

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Check for Iteration marker
            iter_match = line_pattern.search(line)
            if iter_match:
                it_num = int(iter_match.group(1))
                
                # Check for Val Loss (Prioritize val loss capture if on same line)
                val_match = val_pattern.search(line)
                if val_match:
                    val_iterations.append(it_num)
                    val_losses.append(float(val_match.group(1)))

                # Check for Train Loss
                train_match = train_pattern.search(line)
                if train_match:
                    iterations.append(it_num)
                    train_losses.append(float(train_match.group(1)))
                
    return iterations, train_losses, val_iterations, val_losses

def plot_gradient_descent(iterations, train_losses, val_iterations, val_losses, output_file="loss_curve.png"):
    plt.figure(figsize=(12, 7))
    
    # Plot Training Loss
    if iterations and train_losses:
        plt.plot(iterations, train_losses, label='Training Loss', color='#3b82f6', linewidth=1.5, marker='o', markersize=3, alpha=0.7)
        
        # Add a smoothed trend line
        if len(iterations) > 5:
            # Simple moving average
            window = max(2, len(iterations) // 10)
            smoothed = []
            for i in range(len(train_losses)):
                start_idx = max(0, i - window + 1)
                window_slice = train_losses[start_idx : i + 1]
                smoothed.append(sum(window_slice) / len(window_slice))
                
            plt.plot(iterations, smoothed, label='Trend (Moving Avg)', color='#1e3a8a', linewidth=2.5, linestyle='-')

    # Plot Validation Loss
    if val_iterations and val_losses:
        # Highlight start and end
        plt.plot(val_iterations, val_losses, label='Validation Loss', color='#ef4444', linewidth=2, marker='s', markersize=8, zorder=10)
        
        # Annotate Validation points
        for x, y in zip(val_iterations, val_losses):
            plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0,10), ha='center', color='#ef4444', fontweight='bold')

    plt.title('Pirate Adapter Training: Gradient Descent', fontsize=16, fontweight='bold')
    plt.xlabel('Iterations (Steps)', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(fontsize=11)
    
    # Add stats box
    if train_losses:
        start_loss = train_losses[0]
        end_loss = train_losses[-1]
        improvement = ((start_loss - end_loss) / start_loss) * 100
        stats_text = (
            f"Start Train Loss: {start_loss:.3f}\n"
            f"End Train Loss:   {end_loss:.3f}\n"
            f"Improvement:      {improvement:.1f}%"
        )
        plt.text(0.02, 0.05, stats_text, transform=plt.gca().transAxes, 
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='#cccccc'))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"[SUCCESS] Plot saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MLX Training Loss")
    parser.add_argument("--log", type=str, default="MultiLoRa/training_log.txt", help="Path to the training log file")
    parser.add_argument("--out", type=str, default="MultiLoRa/loss_curve.png", help="Path to save the plot image")
    
    args = parser.parse_args()
    
    iters, t_loss, v_iters, v_loss = parse_log(args.log)
    
    if (iters and len(iters) > 0) or (v_iters and len(v_iters) > 0):
        plot_gradient_descent(iters, t_loss, v_iters, v_loss, args.out)
    else:
        print("[WARN] No loss data found in log.")
