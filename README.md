# Wongnai QA System

ระบบถาม-ตอบอัจฉริยะสำหรับค้นหาและแนะนำร้านอาหาร โดยใช้ข้อมูลรีวิวจาก Wongnai ผ่านเทคนิค Semantic Search + Generative QA

## สารบัญ

- [ภาพรวมโปรเจค](#ภาพรวมโปรเจค)
- [สถาปัตยกรรมระบบ](#สถาปัตยกรรมระบบ)
- [Dataset](#dataset)
- [โมเดลที่ใช้](#โมเดลที่ใช้)
- [โครงสร้างโปรเจค](#โครงสร้างโปรเจค)
- [การติดตั้ง](#การติดตั้ง)
- [วิธีใช้งาน](#วิธีใช้งาน)
- [Pipeline ทั้งหมด](#pipeline-ทั้งหมด)
- [รายละเอียดแต่ละโมดูล](#รายละเอียดแต่ละโมดูล)
- [ผลการทดลอง](#ผลการทดลอง)
- [Web UI](#web-ui)
- [ข้อจำกัดและแนวทางพัฒนา](#ข้อจำกัดและแนวทางพัฒนา)

---

## ภาพรวมโปรเจค

Wongnai QA System เป็นระบบ Question Answering ที่สามารถตอบคำถามเกี่ยวกับร้านอาหารจากข้อมูลรีวิว Wongnai จำนวน 40,000 รีวิว โดยรองรับคำถาม 5 ประเภท:

| ประเภทคำถาม | ตัวอย่าง |
|---|---|
| 1. สัญชาติอาหาร (Cuisine) | "ร้านอาหารญี่ปุ่นอร่อยๆ", "อาหารเกาหลีแนะนำ" |
| 2. ประเภทอาหาร (Food Type) | "ร้านบุฟเฟ่ต์ดีๆ", "ชาบูอร่อย" |
| 3. บรรยากาศ/ราคา (Atmosphere/Price) | "ร้านบรรยากาศดี ราคาไม่แพง" |
| 4. สถานที่ (Location) | "ร้านอาหารสยาม", "ร้านอร่อยทองหล่อ" |
| 5. คำถามผสม (Combined) | "ร้านอาหารญี่ปุ่นบรรยากาศดีย่านสยาม" |

### ความสามารถหลัก

- **Semantic Search**: ค้นหารีวิวที่เกี่ยวข้องด้วย Sentence Embedding + FAISS
- **Generative QA**: สร้างคำตอบสรุปจากรีวิวที่ค้นมาได้ (Template-based + LLM-based)
- **Baseline vs Fine-tuned**: เปรียบเทียบผลลัพธ์ระหว่างโมเดลพื้นฐานกับโมเดลที่ fine-tune แล้ว
- **Star Rating**: แสดงคะแนนดาวของแต่ละรีวิว
- **Metadata Extraction**: สกัดข้อมูลประเภทอาหาร, สัญชาติ, บรรยากาศ, ราคา, สถานที่ จากข้อความรีวิว
- **Web UI**: อินเทอร์เฟซสวยงามด้วย Gradio รองรับภาษาไทย

---

## สถาปัตยกรรมระบบ

```
                    ┌─────────────────────────────────┐
                    │         User Query (Thai)        │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │      Query Encoding              │
                    │  (multilingual-e5-base)          │
                    │  prefix: "query: "               │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
   ┌──────────▼─────────┐  ┌──────▼──────────┐         │
   │  Baseline FAISS     │  │  Finetuned FAISS │         │
   │  Index (40K vectors)│  │  Index (40K vec) │         │
   └──────────┬─────────┘  └──────┬──────────┘         │
              │                    │                     │
              └────────┬───────────┘                     │
                       │ Top-K Results                   │
              ┌────────▼────────────┐                    │
              │   QA Generator      │                    │
              │  Template / LLM     │                    │
              └────────┬────────────┘                    │
                       │                                 │
              ┌────────▼────────────┐                    │
              │   Gradio Web UI     │◄───────────────────┘
              │   (3 tabs)          │
              └─────────────────────┘
```

### Data Flow

1. **Preprocessing**: รีวิวดิบ → ทำความสะอาด → สกัด metadata → สร้าง search_text
2. **Indexing**: search_text → Sentence Embedding (768-dim) → FAISS IndexFlatIP
3. **Search**: query → encode → cosine similarity search → Top-K results
4. **Answer Generation**: Top-K results → Template/LLM → คำตอบสรุปภาษาไทย

---

## Dataset

### Wongnai Review Dataset

| ไฟล์ | รายละเอียด | ขนาด |
|---|---|---|
| `w_review_train.csv` | รีวิวร้านอาหาร (review_text;star_rating) | 40,000 รีวิว |
| `food_dictionary.txt` | พจนานุกรมชื่ออาหาร | 409,000 รายการ |
| `labeled_queries_by_judges.txt` | คำค้นหาจาก judge labeling | ~10,000 queries |
| `labeled_queries_by_algo.txt` | คำค้นหาจาก algorithm labeling | ~500,000 queries |
| `test_file.csv` | ชุดทดสอบ | 41,000 รีวิว |

### การกระจายของ Star Rating

| Rating | จำนวน | สัดส่วน |
|---|---|---|
| 1 ดาว | 415 | 1.0% |
| 2 ดาว | 1,845 | 4.6% |
| 3 ดาว | 12,171 | 30.4% |
| 4 ดาว | 18,770 | 46.9% |
| 5 ดาว | 6,799 | 17.0% |

---

## โมเดลที่ใช้

### 1. Embedding Model: `intfloat/multilingual-e5-base`

- **ประเภท**: Sentence Transformer (Multilingual)
- **มิติ**: 768 dimensions
- **ภาษา**: รองรับ 100+ ภาษา รวมถึงไทย
- **ขนาด**: ~1.1 GB
- **หมายเหตุ**: ใช้ prefix "query: " สำหรับ query และ "passage: " สำหรับ document

### 2. QA Model: `scb10x/llama3.1-typhoon2-8b-instruct`

- **ประเภท**: Causal LM (Thai Language Model)
- **ขนาด**: 8B parameters
- **Quantization**: 4-bit NF4 (ผ่าน BitsAndBytesConfig)
- **VRAM ที่ต้องการ**: ~6-8 GB (หลัง quantization)
- **หมายเหตุ**: ใช้สำหรับ LLM-based answer generation, fallback เป็น template mode หาก GPU ไม่พอ

### 3. Fine-tuning Strategy

- **Loss Function**: MultipleNegativesRankingLoss (Contrastive Learning)
- **Training Pairs**: 5,000 คู่ จาก 3 กลยุทธ์:
  - Strategy A (30%): Query-Review pairs จาก labeled queries
  - Strategy B (40%): Similar Review pairs จัดกลุ่มตาม metadata
  - Strategy C (30%): Rating-based pairs (high rating + same food type)
- **Epochs**: 2
- **Batch Size**: 16
- **Warmup Steps**: 100

---

## โครงสร้างโปรเจค

```
Wongnai-QA-System/
├── Dataset/                          # ข้อมูลดิบ
│   ├── review_dataset/
│   │   ├── w_review_train.csv        # รีวิว 40K (semicolon-separated)
│   │   ├── test_file.csv             # ชุดทดสอบ 41K
│   │   └── sample_submission.csv
│   ├── food_dictionary.txt           # พจนานุกรมอาหาร 409K
│   ├── labeled_queries_by_judges.txt # คำค้นหาจาก judge 10K
│   └── labeled_queries_by_algo.txt   # คำค้นหาจาก algo 500K
│
├── src/                              # ซอร์สโค้ดหลัก
│   ├── __init__.py
│   ├── config.py                     # ค่าคอนฟิกทั้งหมด (paths, models, params)
│   ├── data_preprocessing.py         # ทำความสะอาดรีวิว + สกัด metadata
│   ├── retrieval.py                  # Semantic search ด้วย FAISS
│   ├── finetune_embedding.py         # Fine-tune embedding model
│   ├── qa_generator.py               # สร้างคำตอบ (template + LLM)
│   ├── evaluation.py                 # ประเมินและเปรียบเทียบผลลัพธ์
│   └── app.py                        # Gradio Web UI
│
├── data/
│   └── processed/
│       └── processed_reviews.pkl     # รีวิวที่ผ่านการประมวลผลแล้ว (~100 MB)
│
├── models/                           # โมเดลและ index ที่สร้างขึ้น
│   ├── faiss_index                   # FAISS index (baseline) ~123 MB
│   ├── faiss_index_df.pkl            # DataFrame ประกอบ index ~105 MB
│   ├── finetuned_faiss_index         # FAISS index (finetuned) ~123 MB
│   ├── finetuned_faiss_index_df.pkl  # DataFrame ประกอบ index ~105 MB
│   └── finetuned_embedding/          # Fine-tuned embedding model ~1.1 GB
│
├── run_pipeline.py                   # CLI สำหรับรัน pipeline ทั้งหมด
├── requirements.txt                  # Python dependencies
└── README.md                         # เอกสารนี้
```

---

## การติดตั้ง

### ความต้องการระบบ

- Python 3.10+
- CUDA-compatible GPU (แนะนำ VRAM >= 8 GB สำหรับ LLM mode)
- RAM >= 16 GB
- พื้นที่ดิสก์ >= 10 GB (สำหรับโมเดลและ index)

### ขั้นตอนการติดตั้ง

```bash
# 1. Clone โปรเจค
git clone <repository-url>
cd Wongnai-QA-System

# 2. สร้าง Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# หรือ
venv\Scripts\activate     # Windows

# 3. ติดตั้ง Dependencies
pip install -r requirements.txt

# 4. วาง Dataset
# วางโฟลเดอร์ Dataset/ ไว้ในโปรเจคตามโครงสร้างด้านบน
```

### Dependencies หลัก

| Package | เวอร์ชัน | หน้าที่ |
|---|---|---|
| torch | >= 2.0.0 | Deep learning framework |
| transformers | >= 4.30.0 | Model loading, tokenization |
| sentence-transformers | >= 2.2.0 | Sentence embedding + fine-tuning |
| faiss-cpu | >= 1.7.0 | Vector similarity search |
| gradio | >= 4.0.0 | Web UI |
| pandas | >= 2.0.0 | Data manipulation |
| pythainlp | >= 4.0.0 | Thai NLP utilities |
| accelerate | - | Model loading optimization |
| bitsandbytes | - | 4-bit quantization |
| peft | - | Parameter-efficient fine-tuning |

---

## วิธีใช้งาน

### CLI Pipeline (`run_pipeline.py`)

```bash
# รัน pipeline ทั้งหมดตั้งแต่ต้น
python run_pipeline.py all

# หรือรันทีละขั้นตอน:

# 1. Preprocessing - ทำความสะอาดและสกัด metadata
python run_pipeline.py preprocess

# 2. สร้าง FAISS Index (baseline)
python run_pipeline.py build-index

# 3. Fine-tune Embedding Model
python run_pipeline.py finetune

# 4. สร้าง FAISS Index (finetuned)
python run_pipeline.py build-index --finetuned

# 5. ประเมินผลลัพธ์
python run_pipeline.py evaluate

# 6. รัน Demo Queries
python run_pipeline.py demo

# 7. เปิด Web UI
python run_pipeline.py app
```

### เปิด Web UI โดยตรง

```bash
python -m src.app
```

เข้าใช้งานที่ `http://localhost:7860`

### ใช้งานผ่าน Python Code

```python
from src.retrieval import WongnaiRetriever
from src.qa_generator import WongnaiQAGenerator

# โหลด Retriever
retriever = WongnaiRetriever()
retriever.load_index("models/faiss_index", "models/faiss_index_df.pkl")

# ค้นหา
results = retriever.search("ร้านอาหารญี่ปุ่นอร่อย", top_k=5)

# สร้างคำตอบ (template mode)
generator = WongnaiQAGenerator(use_llm=False)
answer = generator.generate_answer("ร้านอาหารญี่ปุ่นอร่อย", results)
print(answer)
```

---

## Pipeline ทั้งหมด

### Step 1: Data Preprocessing (`data_preprocessing.py`)

- อ่านไฟล์ CSV (semicolon-separated, ไม่มี header)
- ทำความสะอาดข้อความ: ลบ URL, ลบอักขระพิเศษ, ลบช่องว่างซ้ำ
- สกัด metadata ด้วย keyword matching:
  - **cuisine_type**: ญี่ปุ่น, เกาหลี, อิตาเลียน, จีน, ไทย ฯลฯ (14 ประเภท)
  - **food_type**: บุฟเฟ่ต์, ชาบู, ซูชิ, กาแฟ, เบเกอรี่ ฯลฯ (23 ประเภท)
  - **atmosphere**: บรรยากาศดี, โรแมนติก, ครอบครัว ฯลฯ (15 ประเภท)
  - **price_level**: ราคาถูก, คุ้มค่า, ราคาแพง ฯลฯ (6 ระดับ)
  - **location**: สยาม, ทองหล่อ, เยาวราช ฯลฯ (6 ย่าน + 38 จังหวัด)
- จับคู่ชื่ออาหารจาก food_dictionary (กรอง 5,000 รายการ, 4-50 ตัวอักษร)
- สร้าง `search_text` = review_text + metadata tags
- บันทึกเป็น `processed_reviews.pkl`
- **เวลาประมวลผล**: ~66 วินาที สำหรับ 40K รีวิว

### Step 2: Build FAISS Index (`retrieval.py`)

- โหลด processed reviews
- Encode ด้วย `intfloat/multilingual-e5-base` (prefix "passage: ")
- L2 Normalize vectors
- สร้าง FAISS IndexFlatIP (Inner Product = Cosine Similarity หลัง normalize)
- **ขนาด Index**: 40,000 vectors x 768 dimensions
- **เวลา**: ~2 นาที 10 วินาที

### Step 3: Fine-tune Embedding (`finetune_embedding.py`)

- สร้าง training pairs 5,000 คู่ จาก 3 กลยุทธ์
- Fine-tune ด้วย `MultipleNegativesRankingLoss`
- บันทึกโมเดลที่ `models/finetuned_embedding/`
- สร้าง FAISS index ใหม่จากโมเดลที่ fine-tune แล้ว
- **เวลา**: ~8 นาที

### Step 4: QA Generation (`qa_generator.py`)

รองรับ 2 โหมด:

| โหมด | รายละเอียด | ข้อดี | ข้อจำกัด |
|---|---|---|---|
| Template | จัดรูปแบบผลลัพธ์เป็นข้อความ Thai | ไม่ต้องใช้ GPU, เสถียร | ไม่ได้สรุปเนื้อหา |
| LLM | ใช้ Typhoon2-8B สรุปคำตอบ | คำตอบเป็นธรรมชาติ | ต้องใช้ GPU >= 8GB VRAM |

### Step 5: Evaluation (`evaluation.py`)

- ทดสอบด้วย 26 demo queries ครอบคลุมทั้ง 5 ประเภท
- วัดผล: avg_retrieval_score, avg_star_rating, metadata_coverage
- เปรียบเทียบ Baseline vs Fine-tuned

---

## รายละเอียดแต่ละโมดูล

### `src/config.py`
ค่าคอนฟิกส่วนกลาง: paths ไฟล์ข้อมูล, ชื่อโมเดล, FAISS paths, TOP_K=5, DEVICE (cuda/cpu)

### `src/data_preprocessing.py`
- `load_reviews()`: อ่าน CSV semicolon-separated
- `load_food_dictionary()`: โหลดพจนานุกรมอาหาร
- `load_queries()`: โหลด labeled queries (pipe-delimited)
- `clean_review()`: ทำความสะอาดข้อความ
- `extract_metadata()`: สกัด metadata ด้วย keyword matching
- `create_search_text()`: สร้าง search text ที่รวม metadata
- `process_all_reviews()`: ประมวลผลรีวิวทั้งหมด

### `src/retrieval.py`
- `WongnaiRetriever`: คลาสหลักสำหรับ semantic search
  - `build_index()`: สร้าง FAISS index จาก DataFrame
  - `load_index()`: โหลด index ที่สร้างไว้แล้ว
  - `search()`: ค้นหาด้วย cosine similarity
  - `search_with_filters()`: ค้นหาพร้อมกรองด้วย metadata

### `src/finetune_embedding.py`
- `generate_training_pairs()`: สร้างคู่ training จาก 3 กลยุทธ์
- `finetune_model()`: Fine-tune ด้วย sentence-transformers .fit()

### `src/qa_generator.py`
- `WongnaiQAGenerator`: คลาส dual-mode answer generation
  - `generate_answer_template()`: สร้างคำตอบแบบ template
  - `generate_answer_llm()`: สร้างคำตอบด้วย Typhoon2-8B
  - `generate_answer()`: Router เลือกโหมดอัตโนมัติ

### `src/evaluation.py`
- `get_demo_queries()`: 26 queries ใน 5 หมวดหมู่
- `evaluate_retrieval()`: วัดผล retrieval quality
- `compare_retrievers()`: เปรียบเทียบ baseline vs finetuned
- `run_demo_evaluation()`: รัน evaluation pipeline เต็มรูปแบบ

### `src/app.py`
- Gradio Blocks UI ภาษาไทย
- 3 แท็บ: ค้นหา, Demo Queries, เกี่ยวกับ
- Custom CSS ด้วยฟอนต์ Sarabun, ธีมสีส้ม-แดง
- รองรับ 3 โหมด: Baseline, Fine-tuned, เปรียบเทียบ

---

## ผลการทดลอง

### Retrieval Score (ตัวอย่าง)

| ประเภทคำถาม | Baseline Score | Fine-tuned Score |
|---|---|---|
| ร้านอาหารญี่ปุ่น | 0.89 | 0.73 |
| ร้านบุฟเฟ่ต์ | 0.87 | 0.68 |
| ร้านบรรยากาศดี | 0.86 | 0.65 |
| ร้านอาหารสยาม | 0.88 | 0.71 |
| ญี่ปุ่น + บรรยากาศดี + สยาม | 0.87 | 0.60 |

**หมายเหตุ**: Baseline score สูงกว่าเนื่องจากคำนวณจาก inner product ของ embedding space เดิม ขณะที่ fine-tuned model มี embedding space ที่ปรับแล้ว ทำให้ค่าคะแนนต่างกัน แต่ผลลัพธ์ที่ดึงมาอาจมีความเกี่ยวข้องเชิงเนื้อหามากกว่า

### เวลาประมวลผล

| ขั้นตอน | เวลา |
|---|---|
| Preprocessing (40K reviews) | ~66 วินาที |
| Build FAISS Index | ~2 นาที 10 วินาที |
| Fine-tuning (5K pairs, 2 epochs) | ~8 นาที |
| Search Query (per query) | < 1 วินาที |

---

## Web UI

เปิดใช้งานที่ `http://localhost:7860` หลังรัน `python run_pipeline.py app`

### แท็บ 1: ค้นหาร้านอาหาร
- พิมพ์คำถามภาษาไทย
- เลือกโหมด: Baseline / Fine-tuned / เปรียบเทียบ
- ปรับจำนวนผลลัพธ์ (1-10)
- แสดงคำตอบสรุป + รีวิวแต่ละร้านพร้อม star rating และ metadata

### แท็บ 2: Demo Queries
- 26 คำค้นหาตัวอย่างพร้อมให้กดทดลอง
- แบ่งตาม 5 หมวดหมู่

### แท็บ 3: เกี่ยวกับระบบ
- รายละเอียดโมเดลและเทคนิคที่ใช้

---

## ข้อจำกัดและแนวทางพัฒนา

### ข้อจำกัดปัจจุบัน

1. **Metadata Over-detection**: keyword matching บางคำจับได้กว้างเกินไป (เช่น "ชา" จับได้ 25K รีวิว เพราะเป็นส่วนของคำอื่น)
2. **LLM Mode**: ต้องการพื้นที่ดิสก์สำหรับดาวน์โหลด Typhoon2-8B (~16 GB) และ VRAM >= 8 GB
3. **CSV Parsing**: รีวิวแบบ multiline ในเครื่องหมายคำพูดอาจพลาดไป ได้ 40K จาก ~247K ที่เป็นไปได้
4. **No Restaurant Name**: Dataset ไม่มีชื่อร้าน ทำให้ไม่สามารถระบุร้านเฉพาะเจาะจงได้

### แนวทางพัฒนาต่อ

1. ปรับปรุง metadata extraction ด้วย Thai word segmentation (pythainlp) เพื่อลด false positive
2. เพิ่มจำนวน training pairs และ epochs สำหรับ fine-tuning
3. ใช้ Cross-encoder reranking เพื่อปรับปรุง retrieval quality
4. เพิ่ม evaluation metrics: MRR, NDCG, Precision@K
5. รองรับ multiline CSV parsing เพื่อใช้ข้อมูลครบ 247K รีวิว

---

## เทคโนโลยีที่ใช้

- **Language**: Python 3.10
- **Deep Learning**: PyTorch, Transformers, Sentence-Transformers
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Thai NLP**: PyThaiNLP
- **LLM**: Typhoon2-8B (SCB10X) with 4-bit quantization
- **Web UI**: Gradio
- **Hardware**: NVIDIA RTX 4070 12GB VRAM

---

## ผู้พัฒนา

โปรเจคนี้เป็นส่วนหนึ่งของวิชา NLP (Natural Language Processing) ระดับปริญญาโท
