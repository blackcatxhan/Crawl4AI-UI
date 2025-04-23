# Crawl4AI

Giao diện người dùng đơn giản cho thư viện Crawl4AI, giúp bạn dễ dàng crawl và trích xuất dữ liệu từ trang web.

## Tính năng chính

- **Crawl cơ bản**: Cào nội dung từ một URL
- **Deep Crawl (BFS)**: Cào sâu nhiều trang sử dụng thuật toán BFS, với giới hạn số trang
- **Trích xuất với CSS Selectors**: Trích xuất dữ liệu có cấu trúc bằng CSS Selectors
- **Trích xuất với LLM**: Trích xuất dữ liệu thông minh sử dụng mô hình ngôn ngữ lớn
- **Hỗ trợ cache**: Tăng tốc độ cào dữ liệu bằng cơ chế cache
- **Lọc nội dung**: Loại bỏ phần không quan trọng trên trang web

## Tính năng

- **Tùy chọn cấu hình**: Kiểm soát cache, ngưỡng nội dung và các tham số khác
- **Giao diện thân thiện**: Dễ sử dụng với Gradio UI

## Cài đặt

1. Clone repository này và vào thư mục dự án:

```bash
git clone https://github.com/blackcatx/crawl4ai-ui.git
cd crawl4ai-ui
```

2. Cài đặt các thư viện phụ thuộc:

```bash
pip install -r requirements.txt
```

3. Cài đặt crawl4ai (nếu chưa có):

```bash
# Install the package
pip install -U crawl4ai

# For pre release versions
pip install crawl4ai --pre

# Run post-installation setup
crawl4ai-setup

# Verify your installation
crawl4ai-doctor
```

4. More details: https://github.com/unclecode/crawl4ai

## Sử dụng

Để chạy ứng dụng:

```bash
python app.py
```

Truy cập giao diện tại: http://localhost:7860

### Hướng dẫn sử dụng

1. **Tab Crawl cơ bản**:
   - Nhập URL cần crawl
   - Chọn chế độ cache (BYPASS để luôn lấy dữ liệu mới)
   - Điều chỉnh ngưỡng nội dung (0.4 là giá trị mặc định tốt)
   - Chọn chế độ trích xuất (markdown, css, llm)
   - Nhấn "Bắt đầu Crawl"
   
2. **Tab Cấu hình CSS**:
   - Nhập schema JSON cho CSS selector
   - Lưu schema trước khi sử dụng chế độ trích xuất CSS
   
3. **Tab Cấu hình LLM**:
   - Chọn nhà cung cấp LLM (OpenAI hoặc Ollama)
   - Nhập API key (chỉ bắt buộc cho OpenAI)
   - Cung cấp schema và hướng dẫn trích xuất
   - Lưu cấu hình trước khi sử dụng chế độ trích xuất LLM

## Yêu cầu hệ thống

- Python 3.9+
- Playwright (được cài đặt tự động bởi Crawl4AI)
- Trình duyệt (Chrome, Firefox hoặc WebKit - được cài đặt bởi lệnh `crawl4ai-setup`)

## Giấy phép

MIT 
