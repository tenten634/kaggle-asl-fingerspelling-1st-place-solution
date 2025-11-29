"""
Calculate training time based on epoch duration
"""
# 1 epoch = 13 minutes 41 seconds
epoch_time_minutes = 13 + (41/60)  # 13.683 minutes
epoch_time_hours = epoch_time_minutes / 60  # 0.228 hours

total_epochs = 400
total_time_hours = epoch_time_hours * total_epochs
total_time_days = total_time_hours / 24

print("="*60)
print("TRAINING TIME CALCULATION")
print("="*60)
print(f"\n⏱️  Per Epoch: {epoch_time_minutes:.2f} minutes ({epoch_time_hours:.3f} hours)")
print(f"📊 Total Epochs: {total_epochs}")
print(f"\n⏰ Total Training Time:")
print(f"   {total_time_hours:.2f} hours")
print(f"   {total_time_days:.2f} days")
print(f"\n⚠️  Kaggle Session Limit: 9 hours")
print(f"   You'll need: {total_time_hours / 9:.1f} sessions minimum")
print(f"\n💡 Options:")
print(f"   1. Save checkpoints every epoch (can resume)")
print(f"   2. Reduce number of epochs")
print(f"   3. Use multiple sessions (save progress between)")
print("="*60)

