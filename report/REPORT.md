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

**Domain:** Customer support / vận hành thương mại điện tử (quy trình đơn hàng, thanh toán, hoàn trả)

**Tại sao nhóm chọn domain này?**
> *Nhóm chọn domain này vì dữ liệu gần với bài toán trợ lý nội bộ và hỗ trợ khách hàng trong thực tế: câu hỏi thường xoay quanh thanh toán, trạng thái đơn hàng, đổi trả và xử lý khiếu nại. Đây cũng là domain có nhiều tình huống tương tự nhau nhưng khác ngữ cảnh, rất phù hợp để kiểm tra chất lượng retrieval theo chunking strategy và metadata filter. Ngoài ra, nhóm có thể dễ xây benchmark query mang tính nghiệp vụ rõ ràng để so sánh giữa các thành viên.*

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `thanhtoan.md` | Tài liệu nghiệp vụ nhóm tự tổng hợp (quy trình thanh toán) | 18,935 | `domain=payment`, `doc_type=policy`, `lang=vi`, `department=cs` |
| 2 | `hoantra.md` | Tài liệu nghiệp vụ nhóm tự tổng hợp (chính sách hoàn trả) | 1,709 | `domain=return_refund`, `doc_type=policy`, `lang=vi`, `department=cs` |
| 3 | `donhang.md` | Tài liệu nghiệp vụ nhóm tự tổng hợp (quản lý đơn hàng) | 912 | `domain=order`, `doc_type=faq`, `lang=vi`, `department=ops` |
| 4 | - | - | - | - |
| 5 | - | - | - | - |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `domain` | string | `payment`, `order`, `return_refund` | Thu hẹp truy xuất theo chủ đề nghiệp vụ, giảm nhiễu giữa các mảng nội dung khác nhau |
| `doc_type` | string | `policy`, `faq` | Tách tài liệu quy định và tài liệu hỏi-đáp để ưu tiên loại nội dung phù hợp mục đích câu hỏi |
| `lang` | string | `vi` | Hữu ích khi hệ thống có dữ liệu đa ngôn ngữ, tránh trả nhầm ngôn ngữ |
| `department` | string | `cs`, `ops` | Cho phép lọc theo phòng ban phụ trách để tăng độ chính xác và khả năng điều phối xử lý |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `thanhtoan.txt` | FixedSizeChunker (`fixed_size`) | 106 | 198.44 | Trung bình: giữ độ dài ổn định nhưng hay cắt ngang ý |
| `thanhtoan.txt` | SentenceChunker (`by_sentences`) | 37 | 510.62 | Khá tốt theo câu, nhưng chunk dài, dễ lẫn nhiều ý |
| `thanhtoan.txt` | RecursiveChunker (`recursive`) | 158 | 118.64 | Tốt: bám cấu trúc đoạn/câu, ngữ cảnh rõ hơn |

### Strategy Của Tôi

**Loại:** RecursiveChunker (tinh chỉnh `chunk_size=250`)

**Mô tả cách hoạt động:**
> *Chiến lược này tách tài liệu theo thứ tự ưu tiên separator: đoạn (`\n\n`) -> dòng (`\n`) -> câu (`. `) -> từ (` `) -> ký tự. Nếu một đoạn vẫn quá dài, thuật toán tiếp tục tách đệ quy bằng separator cấp thấp hơn cho đến khi đạt ngưỡng. Cách làm này giúp phần lớn chunk giữ được ranh giới tự nhiên của văn bản nghiệp vụ thay vì cắt cứng theo số ký tự. Với dữ liệu hướng quy trình như thanh toán, chunk tạo ra thường vừa đủ ngắn để retrieve tốt nhưng vẫn còn đủ ngữ cảnh để trả lời.*

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Domain chăm sóc khách hàng/thanh toán có cấu trúc dạng mục, điều kiện, ngoại lệ và các bước xử lý liên tiếp. Recursive chunking tận dụng tốt cấu trúc này vì ưu tiên tách theo đoạn và câu trước khi buộc phải tách nhỏ hơn. Nhờ đó, kết quả retrieval thường trả về đúng cụm quy trình thay vì các mảnh câu rời rạc.*

**Code snippet (nếu custom):**
```python
from src.chunking import RecursiveChunker

text = './data/thanhtoan.txt'
my_chunker = RecursiveChunker(chunk_size=250)
chunks = my_chunker.chunk(text)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `thanhtoan.txt` | best baseline: `recursive` (chunk_size=200) | 158 | 118.64 | Chính xác cao cho câu hỏi chi tiết, đôi lúc quá nhỏ với ý tổng hợp |
| `thanhtoan.txt` | **của tôi**: `recursive` (chunk_size=250) | 129 | 145.54 | Cân bằng hơn giữa độ chi tiết và đủ ngữ cảnh cho câu trả lời |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Với bộ dữ liệu `thanhtoan.txt`, RecursiveChunker là lựa chọn tốt nhất vì phù hợp cấu trúc tài liệu nghiệp vụ nhiều đoạn và điều kiện. Bản baseline (`chunk_size=200`) có độ chi tiết cao, nhưng cấu hình cá nhân (`chunk_size=250`) cho chunk ít phân mảnh hơn nên thường hữu ích hơn cho bước tổng hợp câu trả lời của agent. Vì vậy, nhóm strategy đề xuất là recursive với tinh chỉnh kích thước chunk theo loại truy vấn.*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Tôi tách câu bằng regex dựa trên các dấu kết thúc phổ biến (`.`, `!`, `?`) kết hợp khoảng trắng để giữ ranh giới ngữ nghĩa tự nhiên. Sau khi tách, mỗi câu được `strip()` và loại phần rỗng để tránh tạo chunk nhiễu. Cuối cùng các câu được gom theo `max_sentences_per_chunk`, giúp output ổn định và dễ kiểm soát độ dài.*

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Thuật toán recursive ưu tiên tách theo separator lớn trước (`\n\n`, `\n`, `. `), sau đó mới giảm dần đến separator nhỏ hơn khi đoạn vẫn vượt `chunk_size`. Base case là khi đoạn hiện tại đã đủ ngắn hoặc không còn separator để tách thêm. Cách này giữ được cấu trúc tài liệu tốt hơn so với cắt cứng theo số ký tự.*

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Trong `add_documents`, tôi chuẩn hóa mỗi tài liệu thành record gồm `id`, `doc_id`, `content`, `metadata`, và `embedding`, sau đó lưu vào store. Ở `search`, hệ thống embed câu hỏi rồi tính điểm tương đồng bằng dot product giữa query vector và từng document vector. Kết quả được sort giảm dần theo `score` và cắt theo `top_k`.*

**`search_with_filter` + `delete_document`** — approach:
> *Tôi lọc metadata trước trong `search_with_filter` để thu nhỏ tập candidate, rồi mới chạy similarity search trên tập đã lọc nhằm tăng precision. Với `delete_document`, store xóa toàn bộ record có `doc_id` tương ứng. Hàm trả về `True` nếu kích thước collection giảm sau thao tác, ngược lại trả `False`.*

### KnowledgeBaseAgent

**`answer`** — approach:
> *`answer()` lấy top-k chunk liên quan từ vector store, ghép thành context rồi chèn vào prompt theo cấu trúc rõ ràng: NGỮ CẢNH -> CÂU HỎI -> TRẢ LỜI. Prompt yêu cầu model chỉ dựa trên context và nói thẳng khi thiếu dữ liệu. Cách inject này giúp hạn chế hallucination và tăng khả năng truy vết nguồn.*

### Test Results

```
# Paste output of: pytest tests/ -v
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Hướng dẫn thanh toán bằng ví điện tử như thế nào? | Tôi muốn trả tiền đơn hàng qua ví điện tử. | high | 0.0696 | Sai |
| 2 | Tôi cần kiểm tra trạng thái đơn hàng ở đâu? | Làm sao theo dõi đơn hàng đang giao? | high | -0.0341 | Sai |
| 3 | Chính sách hoàn tiền mất bao lâu? | Quy trình refund thường bao nhiêu ngày? | high | 0.0501 | Sai |
| 4 | Hướng dẫn tích hợp API thanh toán cho kỹ sư. | Hôm nay trời có mưa lớn ở Hà Nội không? | low | 0.0851 | Sai |
| 5 | Thanh toán thất bại vì OTP hết hạn. | Lỗi OTP khi thanh toán bị quá thời gian xác nhận. | high | -0.1189 | Sai |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Cặp gây bất ngờ nhất là cặp 4 vì hai câu khác domain rõ ràng nhưng điểm vẫn dương. Điều này cho thấy với mock embedding deterministic, similarity score không phản ánh ngữ nghĩa mạnh như embedding model chuyên dụng. Vì vậy khi đánh giá retrieval thực tế, nên dùng embedder phù hợp domain và không kết luận chỉ từ mock score.*

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
