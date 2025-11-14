# SQL Query Generation with LoRA Fine-tuned CodeLlama

This project fine-tunes **CodeLlama-7b-Instruct** using **LoRA (Low-Rank Adaptation)** with **4-bit quantization** to generate SQL queries from natural language questions. The model is trained on the [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) dataset.

## 🎯 Project Overview

- **Base Model**: `codellama/CodeLlama-7b-Instruct-hf`
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit (QLoRA) for efficient training
- **Dataset**: [SQL Create Context Dataset](https://huggingface.co/datasets/b-mc2/sql-create-context)
- **Task**: Natural language to SQL query generation

## 📁 Project Structure

```
sql-lora-project/
├── src/
│   ├── data_preprocessing.py   # Dataset preprocessing and formatting
│   ├── train.py                # LoRA training script
│   ├── evaluate.py             # Model evaluation with metrics
│   └── inference.py            # Interactive and batch inference
├── models/                     # Saved model checkpoints
│   └── checkpoint-*/
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ GPU VRAM for training

### Setup

1. **Clone the repository**:

```bash
git clone https://github.com/kkh0714/SQL-Query-Generation-with-LoRA-Fine-tuned-CodeLlama.git
cd SQL-Query-Generation-with-LoRA-Fine-tuned-CodeLlama.git
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

## 📊 Dataset

The project uses the **SQL Create Context** dataset from HuggingFace:

- **Source**: [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context)
- **Size**: ~78,000 examples
- **Format**: Each example contains:
  - `question`: Natural language query
  - `context`: Database schema (CREATE TABLE statements)
  - `answer`: Target SQL query

## 🔧 Usage

### 1. Data Preprocessing

```bash
python src/data_preprocessing.py
```

This script:

- Loads the SQL dataset from HuggingFace
- Formats data for CodeLlama instruction format
- Splits into train/validation sets
- Tokenizes and prepares for training

### 2. Training

```bash
# the command to train for the project
python src/train.py \
    --base_model codellama/CodeLlama-7b-Instruct-hf \
    --use_4bit \
    --batch_size 2 \
    --max_length 256
```

**Key Training Features**:

- **4-bit quantization** for memory efficiency
- **LoRA adapters** for parameter-efficient fine-tuning
- **Gradient checkpointing** to reduce memory usage
- **Mixed precision training** (FP16/BF16)

**Training Parameters**:

- LoRA rank: 16
- LoRA alpha: 32
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Dropout: 0.05

### 3. Evaluation

Evaluate your trained model on test data:

```bash
python src/evaluate.py \
    --lora_checkpoint ./models/checkpoint-23500 \
    --num_samples 100 \
    --output_file evaluation_results.json
```

**Metrics**:

- **Exact Match Rate**: Percentage of perfectly matched SQL queries
- **SQL Accuracy**: Token-level accuracy between prediction and reference

### 4. Inference

#### Interactive Mode

Chat-like interface for real-time SQL generation:

```bash
python src/inference.py \
    --lora_checkpoint ./models/checkpoint-235000 \
    --use_local
```

#### Single Query

Generate SQL for a specific question:

```bash
python src/inference.py \
    --lora_checkpoint ./models/checkpoint-23500 \
    --question "Show all customers from New York" \
    --context "CREATE TABLE customers (id INT, name VARCHAR(100), city VARCHAR(50))"
```

#### Batch Processing

Process multiple queries from a JSON file:

```bash
python src/inference.py \
    --mode batch \
    --lora_checkpoint ./models/checkpoint-23500 \
    --input_file queries.json \
    --output_file results.json
```

**Input JSON format** (`queries.json`):

```json
[
  {
    "question": "Find all orders from 2024",
    "context": "CREATE TABLE orders (id INT, order_date DATE, amount DECIMAL)"
  },
  {
    "question": "Get top 5 customers by revenue",
    "context": "CREATE TABLE customers (id INT, name VARCHAR, revenue DECIMAL)"
  }
]
```

## 💡 Key Features

✅ **Memory Efficient**: 4-bit quantization enables training on consumer GPUs
 ✅ **Fast Training**: LoRA fine-tunes only ~0.1% of model parameters
 ✅ **High Quality**: CodeLlama base model optimized for code generation
 ✅ **Production Ready**: Includes evaluation metrics and inference scripts
 ✅ **Flexible**: Interactive, single-query, and batch inference modes

## 🛠️ Technical Details

### 4-bit Quantization

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
```

## 📝 Example Results

**Input**:

```
Question: Show me all employees hired in 2023
Schema: CREATE TABLE employees (id INT, name VARCHAR, hire_date DATE, salary DECIMAL)
```

**Generated SQL**:

```sql
SELECT * FROM employees WHERE YEAR(hire_date) = 2023;
```

## 🔍 Troubleshooting

### Out of Memory (OOM) Errors

- Reduce `batch_size` in training
- Enable gradient checkpointing
- Use smaller `max_seq_length`

### Model Downloads

- Use `--use_local` flag to prevent re-downloading
- Models are cached in `~/.cache/huggingface/`

### Slow Inference

- Use `do_sample=False` for faster greedy decoding
- Reduce `max_new_tokens`
- Ensure using GPU (`cuda`)

## 📚 References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [CodeLlama Model](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [SQL Dataset](https://huggingface.co/datasets/b-mc2/sql-create-context)
