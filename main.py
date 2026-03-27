import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import train_model
from evaluation import run_evaluation

def main():
    print("--- Niryo Poker Dice CNN Pipeline ---")
    choice = input("Select mode: [1] Train, [2] Evaluate, [3] Exit: ")
    
    if choice == '1':
        train_model()
    elif choice == '2':
        run_evaluation()
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()