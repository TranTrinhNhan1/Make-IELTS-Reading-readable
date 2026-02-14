from transformers import AutoModelForSeq2SeqLM, AutoConfig
import torch


def load_model(model_checkpoint, device):

    print(f"Tải kiến trúc model: {model_checkpoint}")
    
    # Load Config trước để xem tham số 
    config = AutoConfig.from_pretrained(model_checkpoint)
    
    # Load Model 
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    
    # Chuyển model sang GPU nếu có
    model.to(device)
    
    print("Load model thành công!")
    print_trainable_parameters(model)
    
    return model

def print_trainable_parameters(model):
    
    # In ra số lượng tham số 
    
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    print(f"Model Statistics:")
    print(f"   - Tổng số tham số: {all_param:,}")
    print(f"   - Tham số huấn luyện (Trainable): {trainable_params:,}")
    print(f"   - Tỉ lệ train: {100 * trainable_params / all_param:.2f}%")

# Test 
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("Helsinki-NLP/opus-mt-en-vi", device)