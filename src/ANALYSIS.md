# Phân tích kết quả — Memory Systems for AI Agent

Phần này diễn giải kết quả benchmark offline (deterministic) của hai agent. Số
liệu lấy từ `python benchmark.py`.

## Kết quả

### Standard Benchmark (`data/conversations.json`, 10 hội thoại ngắn)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|-------------------|-------------------------|----------------------|------------------|-----------------------|-------------|
| Baseline | 1614              | 14730                   | 0.00                 | 0.30             | 0                     | 0           |
| Advanced | 2898              | 23316                   | 1.00                 | 1.00             | 265                   | 0           |

### Long-Context Stress Benchmark (`data/advanced_long_context.json`, 1 thread rất dài)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|-------------------|-------------------------|----------------------|------------------|-----------------------|-------------|
| Baseline | 258               | 22180                   | 0.00                 | 0.30             | 0                     | 0           |
| Advanced | 586               | 9161                    | 1.00                 | 1.00             | 209                   | 7           |

## 1. Vì sao Advanced recall tốt hơn Baseline

Baseline chỉ có short-term memory trong cùng một thread. Khi câu hỏi recall được
hỏi trong một thread mới (đúng kịch bản cross-session), baseline không còn ngữ
cảnh nào để dựa vào → recall = 0.

Advanced trích các fact ổn định (tên, nơi ở, nghề nghiệp, đồ uống, style…) và ghi
vào `User.md` theo từng `user_id`. File này tồn tại độc lập với thread, nên ở
thread mới agent vẫn đọc lại được hồ sơ → recall = 1.0.

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn

Ở Standard Benchmark, Advanced tốn nhiều token hơn Baseline (agent tokens
2898 > 1614, prompt tokens 23316 > 14730). Lý do:

- mỗi lượt Advanced kéo theo `User.md` + summary + các message gần nhất vào ngữ
  cảnh, trong khi hội thoại còn ngắn nên baseline chưa kịp phình.
- câu trả lời recap của Advanced dài hơn câu ack ngắn của baseline.

Đây là trade-off cốt lõi: với hội thoại ngắn, chi phí duy trì hệ thống memory
chưa được "hoàn vốn".

## 3. Vì sao compact memory giúp Advanced thắng ở hội thoại dài

Ở Stress Benchmark, mỗi lượt người dùng rất dài. Baseline nạp lại **toàn bộ**
lịch sử mỗi lượt nên `prompt tokens processed` tăng gần như bậc hai theo số lượt
→ 22180.

Advanced kích hoạt compact memory 7 lần: khi token vượt ngưỡng, các message cũ
được nén thành summary có độ dài bị chặn, chỉ giữ vài message gần nhất nguyên
văn. Nhờ đó ngữ cảnh mỗi lượt bị giới hạn → tổng `prompt tokens processed` chỉ
còn 9161 (~2.4× rẻ hơn baseline) mà recall vẫn = 1.0.

Điểm quan trọng: compact tối ưu **chủ yếu ở `prompt tokens processed`**
(ngữ cảnh phải mang theo), không phải ở `agent tokens only`. Đây là lý do compact
"không phải lúc nào cũng thắng": ở hội thoại ngắn, lượng ngữ cảnh tiết kiệm được
quá nhỏ để bù cho chi phí ghi `User.md` và trả lời recap dài hơn.

## 4. Memory file tăng trưởng ra sao và rủi ro đi kèm

`User.md` tăng theo số fact ổn định (265 bytes / 209 bytes ở hai bộ). Trong lab
này tăng trưởng nhỏ vì ta chỉ giữ fact đã được chuẩn hóa thay vì nhật ký nguyên
văn. Rủi ro nếu làm sai:

- nếu lưu mọi câu người dùng nói, file phình vô hạn → mỗi lượt lại tốn ngữ cảnh,
  đúng lỗi mà baseline mắc phải.
- nếu trích sai (ví dụ lấy fact từ một câu hỏi), `User.md` bị nhiễm thông tin rác.
- nếu không xử lý correction, fact cũ sai sẽ tồn tại song song fact mới đúng.

## 5. Bonus đã triển khai (mục tiêu 90–100)

### Confidence gate / question guard
`extract_profile_updates()` bỏ qua các lượt chỉ là câu hỏi (kết thúc bằng `?`
hoặc bắt đầu bằng "bạn biết", "nhắc lại"…). Điều này tránh được lỗi kinh điển:
ghi "tên = gì?" vào hồ sơ khi người dùng hỏi "Mình tên gì?".

- **Giải quyết vấn đề gì:** chống nhiễm `User.md` từ câu hỏi và câu đùa.
- **Cải thiện gì:** trước khi thêm guard, recall standard chỉ ~0.46 vì câu hỏi
  recall ghi đè `name`/`drink`; sau khi thêm guard recall = 1.0.
- **Rủi ro thêm vào:** heuristic có thể bỏ sót fact nằm trong một câu vừa hỏi vừa
  khẳng định; cần test dữ liệu thật để chỉnh ngưỡng.

### Conflict handling (correction-aware)
Các key đơn trị (name, location, profession, drink, food, pet) dùng chế độ
`replace`: correction mới ghi đè giá trị cũ nên không bao giờ tồn tại đồng thời
fact sai. Dataset có sẵn các bẫy: Đà Nẵng → Huế, backend → MLOps, và nhiễu
"product manager (đùa)" / "Hà Nội (đi họp)" — agent giữ đúng giá trị hiện tại.

- **Giải quyết vấn đề gì:** memory bị "đóng băng" ở thông tin lỗi thời.
- **Cải thiện gì:** câu hỏi kiểu "nghề hiện tại là gì" trả về MLOps engineer,
  không phải backend engineer.
- **Rủi ro thêm vào:** ghi đè quá mạnh tay có thể xóa mất lịch sử thay đổi; nếu
  cần audit, nên lưu thêm changelog thay vì chỉ thay tại chỗ.

### Structured entity extraction + append keys
Các key tích lũy (`style`, `interests`) hợp nhất giá trị mới (dedupe) thay vì ghi
đè, để giữ được nhiều preference ổn định cùng lúc.
