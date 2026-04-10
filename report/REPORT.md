# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Trịnh Đức Anh]
**Nhóm:** [C401-A4]
**Ngày:** [10/4/2026]

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *High cosine similarity cho thấy các vector có hướng gần như trùng khít, chứng tỏ sự tương đồng chặt chẽ về mặt nội dung hoặc ngữ nghĩa giữa các đối tượng. Đây là yếu tố cốt lõi giúp các mô hình AI thực hiện tìm kiếm và phân loại dữ liệu chính xác dựa trên bản chất thay vì chỉ so khớp từ khóa đơn thuần.*

**Ví dụ HIGH similarity:**
- Sentence A: "Làm thế nào để tối ưu hóa hiệu suất truy vấn trong cơ sở dữ liệu vector?"
- Sentence B: "Các phương pháp giúp cải thiện tốc độ tìm kiếm trên vector database là gì?"
- Tại sao tương đồng: Dù sử dụng từ ngữ khác nhau (tối ưu hóa vs cải thiện, hiệu suất vs tốc độ), cả hai đều chia sẻ chung một mục tiêu và ngữ cảnh kỹ thuật. Trong các mô hình embedding, chúng sẽ được ánh xạ thành hai vector có hướng gần như trùng khít vì có cùng "tọa độ ngữ nghĩa".

**Ví dụ LOW similarity:**
- Sentence A: "Các kỹ thuật nén mô hình giúp giảm dung lượng file thư viện khi triển khai RAG."
- Sentence B:  "Thời tiết tại Hà Nội hôm nay có mưa rào và dông rải rác vào chiều tối."
- Tại sao khác: Hai câu này thuộc hai miền nội dung hoàn toàn khác biệt (kỹ thuật phần mềm và khí tượng). Các từ khóa và ngữ cảnh không có điểm chung, khiến các vector embedding của chúng hướng về các phía khác nhau trong không gian đa chiều, dẫn đến góc giữa hai vector lớn (gần 90°) và chỉ số Cosine Similarity tiến về mức 0

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Cosine similarity được ưu tiên vì nó tập trung vào hướng (ngữ nghĩa) của vector thay vì độ dài, giúp nhận diện sự tương đồng giữa các văn bản cùng chủ đề bất kể chúng là một câu ngắn hay một đoạn văn dài. Ngược lại, Euclidean distance dễ bị nhiễu bởi quy mô dữ liệu, khiến hai tài liệu có cùng nội dung nhưng chênh lệch số lượng từ bị coi là xa cách về mặt toán học.*

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Document(L) 10,000 ký tự, chunk_size(C)=500, overlap(O)=50. Bước nhảy S = C- O = 50. Số lượng chunk = N = (L-O)/ (C-O) sấp sỉ 22,11 chunk*
> *23 chunks. (Trong đó 22 chunk đầu tiên có kích thước đầy đủ 500 ký tự và chunk cuối cùng chứa phần dư còn lại).*

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Số lượng chunk sẽ tăng lên (từ 23 lên 25 chunks) vì bước nhảy giữa các đoạn ngắn lại, khiến văn bản bị chia nhỏ thành nhiều phần hơn để duy trì phần lặp lại lớn hơn. Việc tăng overlap giúp hạn chế tối đa việc mất ngữ cảnh tại các điểm cắt, đảm bảo các thực thể hoặc ý niệm phức tạp không bị chia tách vụng về giữa hai chunk khác nhau*

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** [ví dụ: Customer support FAQ, Vietnamese law, cooking recipes, ...]

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:*

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| | | | |
| | | | |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| | FixedSizeChunker (`fixed_size`) | | | |
| | SentenceChunker (`by_sentences`) | | | |
| | RecursiveChunker (`recursive`) | | | |

### Strategy Của Tôi

**Loại:** [FixedSizeChunker / SentenceChunker / RecursiveChunker / custom strategy]

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?*

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?*

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| | best baseline | | | |
| | **của tôi** | | | |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu: dùng regex gì để detect sentence? Xử lý edge case nào?*

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu: algorithm hoạt động thế nào? Base case là gì?*

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu: lưu trữ thế nào? Tính similarity ra sao?*

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu: filter trước hay sau? Delete bằng cách nào?*

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: prompt structure? Cách inject context?*

### Test Results

```
# Paste output of: pytest tests/ -v
```

**Số tests pass:** __ / __

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | | | high / low | | |
| 2 | | | high / low | | |
| 3 | | | high / low | | |
| 4 | | | high / low | | |
| 5 | | | high / low | | |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:*

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
