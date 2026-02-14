import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

class IELTSTranslationDataset:
    def __init__(self, model_checkpoint, max_length=128, sample_size=None):
        self.model_checkpoint = model_checkpoint
        self.max_length = max_length
        self.sample_size = sample_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        
        self.source_lang = "en" 
        self.target_lang = "vi"

    def preprocess_function(self, examples):

        raw_inputs = examples[self.source_lang]
        raw_targets = examples[self.target_lang]

        # Chuyển câu thành list và bỏ các dòng trống
        inputs = [str(x) if x is not None else "" for x in raw_inputs]
        targets = [str(x) if x is not None else "" for x in raw_targets]

        
        # tokenize input tiếng anh
        model_inputs = self.tokenizer(
            inputs, 
            max_length=self.max_length, 
            truncation=True,
            padding="max_length"
        )

        # tokenize ouput tiếng việt
        labels = self.tokenizer(
            text_target=targets,
            max_length=self.max_length, 
            truncation=True, 
            padding="max_length"
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def load_data(self):
        print(f"Đang tải dataset hiimbach/mtet...")
        raw_dataset = load_dataset("hiimbach/mtet", split="train")
        
        # gán tên cho cột
        self.source_lang = "en"
        self.target_lang = "vi"

        print(f"Đầu vào là: '{self.source_lang}', Đầu ra là: '{self.target_lang}'")

        # Chọn ra sample_size câu để train 
        if self.sample_size:
            print(f"Đang lấy mẫu ngẫu nhiên {self.sample_size} câu...")
            raw_dataset = raw_dataset.shuffle(seed=42).select(range(self.sample_size))

        # Vì dữ liệu tải về chỉ là một duy nhất (Có thể tham khảo notebook) nên cần chia lại thành tập train, validation và test
        # Lấy tỉ lệ 80/10/10
        train_test_split = raw_dataset.train_test_split(test_size=0.2, seed=42)
        test_valid_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)

        dataset = DatasetDict({
            'train': train_test_split['train'],
            'validation': test_valid_split['train'],
            'test': test_valid_split['test']
        })

        # Sau khi chia dữ liệu xong ta bắt đầu tokenize dữ liệu và dùng hàm map để ánh xạ cho cả tập dữ liệu
        print("Đang Tokenize dữ liệu...")
        tokenized_datasets = dataset.map(
            self.preprocess_function, 
            batched=True,
            remove_columns=dataset["train"].column_names 
        )
        
        print("Đã xử lý dữ liệu xong.")
        return tokenized_datasets, self.tokenizer

if __name__ == "__main__":
    # Kiểm tra nhanh một chút
    dataset_handler = IELTSTranslationDataset("Helsinki-NLP/opus-mt-en-vi", sample_size=100)
    data, tokenizer = dataset_handler.load_data()
    print("\nKiểm tra mẫu sau khi xử lý:")
    print(data['train'][0])