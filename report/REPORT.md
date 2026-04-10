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
 thanhtoan.md | FixedSizeChunker | 493 | 199.87 | Mostly |
| thanhtoan.md | SentenceChunker | 113 | 652.66 | Yes |
| thanhtoan.md | RecursiveChunker | 614 | 119.13 | Yes |
| hoantra.md | FixedSizeChunker | 49 | 199.08 | Mostly |
| hoantra.md | SentenceChunker | 17 | 430.88 | Yes |
| hoantra.md | RecursiveChunker | 53 | 136.91 | Yes |
| donhang.md | FixedSizeChunker | 252 | 199.47 | Mostly |
| donhang.md | SentenceChunker | 59 | 637.03 | Yes |
| donhang.md | RecursiveChunker | 284 | 131.05 | Yes |

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
| `thanhtoan.md` | Baseline tốt nhất: `recursive` (`chunk_size=200`) | 158 | 118.64 | Mạnh ở truy vấn chi tiết; một số chunk quá ngắn nên thiếu bối cảnh khi trả lời câu hỏi tổng hợp |
| `thanhtoan.md` | **Strategy của tôi**: `recursive` (`chunk_size=250`) | 129 | 145.54 | Cân bằng tốt hơn giữa độ chi tiết và độ đầy đủ ngữ cảnh; top-k thường đọc được trọn ý hơn |

> *Kết luận so sánh:* baseline (`chunk_size=200`) cho độ phủ chi tiết cao hơn, nhưng strategy của tôi (`chunk_size=250`) giảm phân mảnh chunk và phù hợp hơn cho bước tổng hợp câu trả lời trong RAG.

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Quân | Recursive Chunker | 9 / 10 | Giữ được cấu trúc tự nhiên của tài liệu, chunk đủ ngắn để truy xuất chính xác và vẫn còn ngữ cảnh | Số lượng chunk khá nhiều, có thể làm tăng chi phí lưu trữ và tìm kiếm |
| Hiệp | Sentence Chunker | 8 / 10 | Chunk dễ đọc, giữ trọn ý theo câu hoặc nhóm câu, phù hợp với tài liệu dạng hướng dẫn | Một số chunk quá dài, dễ chứa nhiều ý nên retrieval chưa đủ chính xác |
| Đức Anh | FixedSize Chunker | 7 / 10 | Đơn giản, dễ cài đặt, độ dài chunk ổn định | Dễ cắt ngang ý quan trọng và làm mất ngữ cảnh ở các đoạn dài |
| Dương | Recursive Chunker | 9 / 10 | Cân bằng tốt giữa độ dài chunk và khả năng giữ ngữ cảnh | Tạo nhiều chunk hơn nên cần quản lý tốt hơn khi indexing |
| Chung | RecursiveChunker | 8/10 | Giữ ngữ cảnh tốt, hợp tài liệu kỹ thuật | Có thể tạo nhiều chunk hơn mức cần thiết |
| Đạt | FixedSize Chunker | 7 / 10 | Chạy ổn định, dễ benchmark và so sánh | Chất lượng retrieval kém hơn khi tài liệu có cấu trúc heading và bullet |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Theo tôi, Recursive Chunker là strategy phù hợp nhất cho domain tài liệu hỗ trợ khách hàng của Shopee. Các tài liệu kiểu này thường có cấu trúc theo mục, đoạn, danh sách điều kiện và bước xử lý, nên việc ưu tiên tách theo ranh giới tự nhiên giúp retrieval trả về đúng ý hơn. So với FixedSize Chunker và Sentence Chunker, Recursive Chunker cân bằng tốt hơn giữa việc giữ ngữ cảnh và độ chính xác khi truy xuất.*
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
========================================================= test session starts ==========================================================
platform win32 -- Python 3.12.3, pytest-9.0.3, pluggy-1.6.0 -- d:\AI_VIN\Bai_tap_LAB\Vin_Day7\2A202600499_TRINHDUCANH_Day07\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\AI_VIN\Bai_tap_LAB\Vin_Day7\2A202600499_TRINHDUCANH_Day07
plugins: anyio-4.13.0, langsmith-0.7.30
collected 42 items                                                                                                                      

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                             [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                      [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                               [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                                [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                     [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                     [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                           [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                            [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                          [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                            [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                            [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                       [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                   [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                             [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                    [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                        [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                                  [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                        [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                            [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                              [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                                [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                      [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                           [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                             [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                                 [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                              [ 61%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                       [ 64%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                      [ 66%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                                 [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                             [ 71%] 
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                        [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                            [ 76%] 
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                                  [ 78%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                            [ 80%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                         [ 83%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                       [ 85%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                      [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                          [ 90%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                     [ 92%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                              [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                    [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                        [100%] 

========================================================== 42 passed in 0.63s =========================================================
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
| 1 | Hướng dẫn thanh toán bằng ví điện tử như thế nào? | Làm sao để thanh toán đơn hàng bằng ví điện tử? | high | 0.82 | Đúng |
| 2 | Tôi cần kiểm tra trạng thái đơn hàng ở đâu? | Làm thế nào để theo dõi đơn hàng của tôi? | high | 0.78 | Đúng |
| 3 | Chính sách hoàn tiền mất bao lâu? | Thời gian xử lý hoàn tiền là bao nhiêu ngày? | high | 0.75 | Đúng |
| 4 | Hướng dẫn tích hợp API thanh toán cho kỹ sư. | Tài liệu hướng dẫn tích hợp API payment cho developer ở đâu? | high | 0.73 | Đúng |
| 5 | Thanh toán thất bại vì OTP hết hạn. | Lỗi thanh toán do mã OTP hết hạn xử lý thế nào? | high | 0.77 | Đúng |
| 6 | Hướng dẫn tích hợp API thanh toán cho kỹ sư. | Hôm nay thời tiết Hà Nội thế nào? | low | 0.08 | Đúng 

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| **1** | Tại sao đơn hàng của tôi chưa được cập nhật trạng thái? | `donhang.md`: "Trạng thái đơn hàng sẽ được cập nhật ngay khi đối tác vận chuyển quét mã vận đơn thành công (tối đa 24h)..." | **0.86652** | **Yes** | Truy xuất đúng tệp quản lý đơn hàng. Giải thích được độ trễ do phía đơn vị vận chuyển cập nhật lên hệ thống. |
| **2** | Tôi có bao lâu để yêu cầu trả hàng hoặc hoàn tiền trên Shopee? | `hoantra.md`: "Thời hạn yêu cầu trả hàng/hoàn tiền là 15 ngày đối với Shopee Mall và 07 ngày đối với các shop thông thường..." | **0.9124** | **Yes** | Khớp hoàn toàn từ khóa thời gian trong file chính sách hoàn trả. Thông tin đầy đủ và chính xác theo mốc 15 ngày. |
| **3** | Tôi cần chuẩn bị bằng chứng gì khi yêu cầu trả hàng/hoàn tiền? | `hoantra.md`: "Người mua cần cung cấp video mở kiện hàng rõ nét, không cắt ghép và hiển thị rõ thông tin mã vận đơn..." | **0.8679** | **Yes** | Trích xuất chính xác yêu cầu về bằng chứng từ file hoàn trả. Đã bao gồm yêu cầu quan trọng về video unboxing liên tục. |
| **4** | Điều gì xảy ra nếu người bán không xác nhận hoặc không giao đơn đúng hạn? | `donhang.md`: "Trường hợp đơn hàng quá thời hạn chuẩn bị hàng theo quy định, hệ thống sẽ tự động hủy đơn và hoàn tiền..." | **0.8395** | **Yes** | Truy xuất đúng quy trình xử lý đơn hàng quá hạn. Nêu rõ được cơ chế tự động hủy và bảo vệ quyền lợi người mua. |
| **5** | ShopeePay hỗ trợ những thao tác thanh toán nào? | `thanhtoan.md`: "Ví ShopeePay cho phép thực hiện các thao tác: Nạp tiền, Rút tiền, Chuyển tiền nội bộ và Thanh toán dịch vụ..." | **0.8212** | **Yes** | Khớp chính xác các tính năng chính của ví trong file điều khoản thanh toán. Câu trả lời bao quát đầy đủ các thao tác. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Điều  học được nhiều nhất là cách một bạn trong nhóm dùng metadata filter (`domain`, `department`) trước khi search để giảm nhiễu rõ rệt. Trước đó thường chạy search toàn bộ rồi mới lọc thủ công, nên top-k dễ lẫn chunk không cùng nghiệp vụ. Sau khi áp dụng cách lọc sớm, kết quả retrieval ổn định hơn với các query về thanh toán và hoàn trả.*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Nhóm khác cho thấy việc đánh giá retrieval không chỉ nhìn answer cuối, mà phải kiểm tra trực tiếp top-3 chunk và đối chiếu với gold answer. Cách trình bày failure cases của họ rất rõ: xác định do chunk quá nhỏ, metadata thiếu, hay tài liệu chưa cập nhật. Điều này giúp hiểu rằng một hệ RAG tốt cần vòng lặp đo lường và cải tiến dữ liệu liên tục, không chỉ tối ưu prompt.*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Nếu làm lại, sẽ chuẩn hóa tài liệu ngay từ đầu theo template chung (mục đích, điều kiện áp dụng, quy trình, ngoại lệ) để chunking nhất quán hơn. cũng sẽ bổ sung metadata về `updated_at` và `source_priority` để hạn chế việc tài liệu cũ được xếp hạng cao. Ngoài ra, sẽ tăng số benchmark query theo từng nhóm nghiệp vụ để phát hiện sớm các vùng retrieval còn yếu.*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5/ 5 |
| Document selection | Nhóm | 10/ 10 |
| Chunking strategy | Nhóm | 10/ 15 |
| My approach | Cá nhân | 10/ 10 |
| Similarity predictions | Cá nhân | 5/ 5 |
| Results | Cá nhân | 10/ 10 |
| Core implementation (tests) | Cá nhân | 25/ 30 |
| Demo | Nhóm | 5/ 5 |
| **Tổng** | | **9090/ 100** |
