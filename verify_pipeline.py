"""
Verify the full pipeline works before training.
Tests: imports, data loading, model creation, single forward pass.
"""
import torch
import sys

print("=" * 60)
print("PIPELINE VERIFICATION")
print("=" * 60)

# 1. Test imports
print("\n[1/5] Testing imports...")
from src.config import DataConfig, TeacherConfig, StudentConfig, DistillationConfig
from src.data.dataset import create_dataloaders, get_label_map
from src.models.teacher import TeacherModel
from src.models.student import StudentModel
from src.training.train_teacher import train_teacher
from src.training.train_student import train_student, DistillationLoss
from src.training.logger import TrainingLogger
from src.evaluation.evaluate import evaluate_model
print("  ✓ All imports successful")

# 2. Test data loading
print("\n[2/5] Testing data loading...")
data_config = DataConfig()
label_map = get_label_map(data_config.data_dir)
print(f"  Labels: {label_map}")

train_loader, val_loader, test_loader = create_dataloaders(
    data_dir=data_config.data_dir,
    tokenizer_name=data_config.tokenizer_name,
    max_length=data_config.max_length,
    batch_size=32,
)
print(f"  Train: {len(train_loader)} batches")
print(f"  Val: {len(val_loader)} batches")
print(f"  Test: {len(test_loader)} batches")

# Get one batch
batch = next(iter(train_loader))
print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
print(f"  Batch attention_mask shape: {batch['attention_mask'].shape}")
print(f"  Batch labels shape: {batch['label'].shape}")
assert batch['input_ids'].shape[1] == 32, f"Expected max_length=32, got {batch['input_ids'].shape[1]}"
print("  ✓ Data loading works, max_length=32 confirmed")

# 3. Test teacher model
print("\n[3/5] Testing teacher model...")
teacher = TeacherModel(num_labels=7)
print(f"  Parameters: {teacher.get_num_parameters():,}")
print(f"  Size: {teacher.get_model_size_mb():.2f} MB")

# Forward pass
with torch.no_grad():
    outputs = teacher(batch['input_ids'], batch['attention_mask'], batch['label'])
print(f"  Logits shape: {outputs['logits'].shape}")
print(f"  Loss: {outputs['loss'].item():.4f}")
print("  ✓ Teacher forward pass works")

# 4. Test student model
print("\n[4/5] Testing student model...")
student_config = StudentConfig()
student = StudentModel(
    num_labels=7,
    hidden_size=student_config.hidden_size,
    num_layers=student_config.num_layers,
    num_heads=student_config.num_heads,
    intermediate_size=student_config.intermediate_size,
    max_length=data_config.max_length,
)
print(f"  Parameters: {student.get_num_parameters():,}")
print(f"  Size: {student.get_model_size_mb():.2f} MB")
print(f"  Compression ratio: {teacher.get_num_parameters()/student.get_num_parameters():.1f}x")

# Forward pass
with torch.no_grad():
    outputs = student(batch['input_ids'], batch['attention_mask'], batch['label'])
print(f"  Logits shape: {outputs['logits'].shape}")
print(f"  Loss: {outputs['loss'].item():.4f}")
print("  ✓ Student forward pass works")

# 5. Test distillation loss
print("\n[5/5] Testing distillation loss...")
distill_loss = DistillationLoss(temperature=4.0, alpha=0.7)
with torch.no_grad():
    teacher_out = teacher(batch['input_ids'], batch['attention_mask'])
    student_out = student(batch['input_ids'], batch['attention_mask'])
    loss_dict = distill_loss(student_out['logits'], teacher_out['logits'], batch['label'])
print(f"  Total loss: {loss_dict['loss'].item():.4f}")
print(f"  Distill loss: {loss_dict['distill_loss']:.4f}")
print(f"  CE loss: {loss_dict['ce_loss']:.4f}")
print("  ✓ Distillation loss works")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED — Pipeline ready for training!")
print("=" * 60)
print(f"\nTo train teacher: python -m src.training.train_teacher")
print(f"To train student: python -m src.training.train_student --mode baseline")
print(f"To distill:       python -m src.training.train_student --mode distill")
