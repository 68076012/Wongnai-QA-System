# TODO - Wongnai QA System

## สถานะปัจจุบัน (2026-03-01)

ระบบ QA ทำงานได้ครบ: preprocessing, retrieval, fine-tuning, evaluation, web UI
เหลืองานเอกสารและปรับปรุงเพิ่มเติม

---

## งานที่เหลือ (เรียงตามความสำคัญ)

### 1. รายงาน (Report Document)
**สถานะ**: ยังไม่เริ่ม
**ความสำคัญ**: สูงมาก

เนื้อหาที่ต้องครอบคลุม:
- [ ] บทที่ 1: บทนำ - ที่มาและความสำคัญ, วัตถุประสงค์, ขอบเขต
- [ ] บทที่ 2: ทฤษฎีที่เกี่ยวข้อง
  - [ ] Sentence Transformers & Multilingual Embeddings
  - [ ] FAISS และ Vector Similarity Search
  - [ ] Contrastive Learning (MultipleNegativesRankingLoss)
  - [ ] Retrieval-Augmented Generation (RAG)
  - [ ] 4-bit Quantization (BitsAndBytesConfig)
  - [ ] E5 Model Architecture และ prefix convention
- [ ] บทที่ 3: วิธีการดำเนินงาน
  - [ ] Dataset description และสถิติ
  - [ ] Data preprocessing pipeline (cleaning, metadata extraction)
  - [ ] Baseline retrieval system (e5-base + FAISS)
  - [ ] Fine-tuning strategy (3 strategies, training pairs)
  - [ ] QA generation (template vs LLM)
  - [ ] System architecture diagram
- [ ] บทที่ 4: ผลการทดลอง
  - [ ] Retrieval score comparison (baseline vs finetuned)
  - [ ] ตัวอย่างผลลัพธ์แต่ละประเภทคำถาม (5 ประเภท)
  - [ ] Screenshot ของ Web UI
  - [ ] ตารางเปรียบเทียบ metrics
  - [ ] วิเคราะห์ว่าทำไม baseline score สูงกว่า finetuned
- [ ] บทที่ 5: สรุปและข้อเสนอแนะ
  - [ ] สรุปผล
  - [ ] ข้อจำกัด
  - [ ] แนวทางพัฒนาต่อ
- [ ] เอกสารอ้างอิง (References)
- [ ] ภาคผนวก: ซอร์สโค้ดสำคัญ

### 2. สไลด์นำเสนอ (PPT Presentation)
**สถานะ**: ยังไม่เริ่ม
**ความสำคัญ**: สูงมาก

เนื้อหาสไลด์:
- [ ] สไลด์ 1: หน้าปก - ชื่อโปรเจค, ชื่อผู้จัดทำ
- [ ] สไลด์ 2-3: บทนำ - ปัญหาและวัตถุประสงค์
- [ ] สไลด์ 4-5: Dataset - Wongnai review dataset, สถิติ, ตัวอย่าง
- [ ] สไลด์ 6-7: สถาปัตยกรรมระบบ - Architecture diagram, data flow
- [ ] สไลด์ 8: Data Preprocessing - cleaning, metadata extraction
- [ ] สไลด์ 9-10: Retrieval System - e5-base, FAISS, cosine similarity
- [ ] สไลด์ 11-12: Fine-tuning - contrastive learning, 3 strategies, training pairs
- [ ] สไลด์ 13: QA Generation - template vs LLM mode
- [ ] สไลด์ 14-16: ผลการทดลอง - ตาราง, กราฟ, ตัวอย่างผลลัพธ์
- [ ] สไลด์ 17: Demo - screenshot ของ Web UI
- [ ] สไลด์ 18: สรุปและข้อเสนอแนะ
- [ ] สไลด์ 19: Q&A

### 3. วิดีโอนำเสนอ (Video Presentation)
**สถานะ**: ยังไม่เริ่ม
**ความสำคัญ**: สูง

- [ ] เตรียมสคริปต์การนำเสนอ
- [ ] อัดหน้าจอ demo ของ Web UI
  - [ ] แสดงการค้นหาแต่ละประเภท (5 ประเภท)
  - [ ] แสดงการเปรียบเทียบ baseline vs finetuned
  - [ ] แสดง demo queries tab
- [ ] อัดเสียงอธิบายพร้อมสไลด์
- [ ] ตัดต่อวิดีโอ

---

## งานปรับปรุงเพิ่มเติม (Nice-to-have)

### 4. ปรับปรุง Metadata Extraction
**ความสำคัญ**: ปานกลาง

- [ ] ใช้ pythainlp word segmentation แทน substring matching เพื่อลด false positive
- [ ] แก้ "ชา" (tea) ที่จับได้ 25K รีวิว → ใช้ word boundary
- [ ] แก้ "น่าน" (Nan province) ที่จับผิด → ใช้ word boundary
- [ ] เพิ่ม keyword ใหม่ตามที่พบจากรีวิว

### 5. ปรับปรุง Retrieval Quality
**ความสำคัญ**: ปานกลาง

- [ ] เพิ่มจำนวน training pairs (10K-20K)
- [ ] เพิ่ม epochs (3-5)
- [ ] ทดลอง Cross-encoder reranking (ms-marco-MiniLM)
- [ ] เพิ่ม evaluation metrics: MRR, NDCG, Precision@K, Recall@K
- [ ] สร้าง ground truth test set สำหรับ quantitative evaluation

### 6. ปรับปรุง CSV Parsing
**ความสำคัญ**: ต่ำ

- [ ] แก้ multiline CSV parsing เพื่อใช้ข้อมูลครบ 247K รีวิว
- [ ] ทดสอบผลลัพธ์กับ dataset ขนาดใหญ่ขึ้น

### 7. ปรับปรุง Web UI
**ความสำคัญ**: ต่ำ

- [ ] เพิ่ม filter UI สำหรับ cuisine type, location, price level
- [ ] เพิ่มกราฟเปรียบเทียบ (bar chart / radar chart)
- [ ] เพิ่ม loading animation
- [ ] เพิ่ม export ผลลัพธ์เป็น CSV/PDF

### 8. LLM Mode
**ความสำคัญ**: ต่ำ (ต้องมี disk space + GPU)

- [ ] ดาวน์โหลด Typhoon2-8B ให้สำเร็จ (ต้องการ ~16 GB disk)
- [ ] ทดสอบ LLM-based answer generation
- [ ] เปรียบเทียบคุณภาพคำตอบ template vs LLM

---

## บันทึก

### สิ่งที่ทำเสร็จแล้ว
- [x] Project structure setup
- [x] Data preprocessing pipeline (40K reviews, 66s)
- [x] Baseline retrieval system (FAISS 40K vectors, 2m10s)
- [x] Fine-tune embedding model (5K pairs, 2 epochs, 8m)
- [x] Build finetuned FAISS index
- [x] QA generator (template + LLM dual mode)
- [x] Evaluation module (26 demo queries, 5 categories)
- [x] Gradio Web UI (Thai, 3 tabs, beautiful design)
- [x] CLI pipeline script (run_pipeline.py)
- [x] README.md

### Hardware ที่ใช้
- GPU: NVIDIA RTX 4070 12GB VRAM
- OS: Windows 11 Pro
- Python: 3.10
