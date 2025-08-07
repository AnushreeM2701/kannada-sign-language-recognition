#!/usr/bin/env python3
"""
Comprehensive training script for enhanced Kannada sign language recognition
Uses fixed versions of alphabet and word training scripts
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_alphabet_training import EnhancedAlphabetTrainer

def main():
    parser = argparse.ArgumentParser(description='Train enhanced Kannada sign language models')
    parser.add_argument('--alphabet-only', action='store_true', help='Train only alphabet model')
    parser.add_argument('--word-only', action='store_true', help='Train only word model')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    parser.add_argument('--model-dir', default='models', help='Model directory path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    
    args = parser.parse_args()
    
    print("🚀 Kannada Sign Language Enhanced Training - Fixed Version")
    print("=" * 55)
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📁 Model directory: {args.model_dir}")
    print(f"📊 Epochs: {args.epochs}")
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Training summary
    training_summary = {
        "started_at": str(datetime.now()),
        "data_dir": args.data_dir,
        "model_dir": args.model_dir,
        "epochs": args.epochs,
        "version": "fixed_v2"
    }
    
    # Train alphabet model
    if not args.word_only:
        print("\n📚 Training Enhanced Alphabet Model...")
        try:
            alphabet_trainer = EnhancedAlphabetTrainer(
                data_dir=os.path.join(args.data_dir, "alphabet_images"),
                model_dir=args.model_dir
            )
            alphabet_model = alphabet_trainer.train()
            training_summary["alphabet_training"] = {
                "status": "completed",
                "completed_at": str(datetime.now())
            }
            print("✅ Alphabet training completed successfully!")
        except Exception as e:
            training_summary["alphabet_training"] = {
                "status": "failed",
                "error": str(e)
            }
            print(f"❌ Alphabet training failed: {e}")
    
    # Train word model - Note: word training module doesn't exist
    if not args.alphabet_only:
        print("\n🎬 Word model training skipped - word training module not found")
        training_summary["word_training"] = {
            "status": "skipped",
            "reason": "word training module not found"
        }
    
    # Save training summary
    summary_path = os.path.join(args.model_dir, 'enhanced_training_summary_fixed_v2.json')
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2, ensure_ascii=False)
    
    print("\n🎉 Enhanced training complete - Fixed Version!")
    print(f"📋 Training summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
