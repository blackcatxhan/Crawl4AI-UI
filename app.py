import os
import json
import asyncio
import gradio as gr
import time
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy, JsonCssExtractionStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from dotenv import load_dotenv

# Tải biến môi trường từ file .env (nếu có)
load_dotenv()

# Biến toàn cục cho crawler để tái sử dụng
crawler_instance = None
loop = None

class CrawlApp:
    def __init__(self):
        self.browser_config = BrowserConfig(headless=True)
        
    async def init_crawler(self):
        """Khởi tạo và trả về crawler nếu chưa tồn tại"""
        global crawler_instance
        if crawler_instance is None:
            crawler_instance = AsyncWebCrawler(config=self.browser_config)
            await crawler_instance.start()
        return crawler_instance
        
    async def crawl_url(self, url, cache_mode="BYPASS", content_threshold=0.1, extraction_mode="markdown"):
        """Hàm thực hiện crawl theo URL và cấu hình"""
        # Tạo bộ lọc nội dung và trình tạo markdown
        content_filter = PruningContentFilter(threshold=float(content_threshold), threshold_type="fixed")
        md_generator = DefaultMarkdownGenerator(content_filter=content_filter)
        
        # Cấu hình CrawlerRunConfig cơ bản
        run_config = CrawlerRunConfig(
            cache_mode=getattr(CacheMode, cache_mode),
            markdown_generator=md_generator
        )
        
        # Thêm chiến lược trích xuất nếu cần
        if extraction_mode == "css" and hasattr(self, 'css_schema') and self.css_schema:
            try:
                schema = json.loads(self.css_schema)
                run_config = run_config.clone(
                    extraction_strategy=JsonCssExtractionStrategy(schema)
                )
            except Exception as e:
                return f"Lỗi schema CSS: {str(e)}", None, "{}", None
        elif extraction_mode == "llm" and hasattr(self, 'llm_schema') and self.llm_schema:
            # Chỉ sử dụng LLM với API key hoặc mô hình cục bộ
            provider = self.llm_provider if hasattr(self, 'llm_provider') else "ollama/llama3.3"
            api_key = self.llm_api_key if hasattr(self, 'llm_api_key') else None
            
            if (provider.startswith("openai") and not api_key):
                return "Cần API key cho OpenAI", None, "{}", None
                
            try:
                schema = json.loads(self.llm_schema)
                instruction = self.llm_instruction if hasattr(self, 'llm_instruction') else "Trích xuất dữ liệu theo schema"
                
                run_config = run_config.clone(
                    extraction_strategy=LLMExtractionStrategy(
                        llm_config=LLMConfig(provider=provider, api_token=api_key),
                        schema=schema,
                        extraction_type="schema",
                        instruction=instruction,
                        extra_args={"temperature": 0, "top_p": 0.9}
                    )
                )
            except Exception as e:
                return f"Lỗi schema LLM: {str(e)}", None, "{}", None
        
        # Thực hiện crawl
        start_time = time.time()
        try:
            # Lấy instance crawler hiện có hoặc tạo mới
            crawler = await self.init_crawler()
            
            # Thực hiện crawl
            result = await crawler.arun(url=url, config=run_config)
            duration = time.time() - start_time
            
            # Hiển thị thông tin về việc sử dụng cache
            cache_info = "từ cache" if cache_mode == "ENABLED" and duration < 0.5 else "mới crawl"
            
            # Kiểm tra và đảm bảo kết quả đúng khi dùng cache
            if hasattr(result, 'markdown'):
                # Chuẩn bị kết quả trả về
                if hasattr(result.markdown, 'fit_markdown'):
                    markdown_content = result.markdown.fit_markdown
                elif hasattr(result.markdown, 'raw_markdown'):
                    markdown_content = result.markdown.raw_markdown
                else:
                    markdown_content = str(result.markdown)
                
                # Đảm bảo markdown_content không trống
                if not markdown_content or markdown_content.strip() == "":
                    if cache_mode == "ENABLED":
                        # Thử crawl lại với BYPASS nếu kết quả từ cache trống
                        bypass_config = run_config.clone(cache_mode=CacheMode.BYPASS)
                        result = await crawler.arun(url=url, config=bypass_config)
                        if hasattr(result.markdown, 'fit_markdown'):
                            markdown_content = result.markdown.fit_markdown
                        elif hasattr(result.markdown, 'raw_markdown'):
                            markdown_content = result.markdown.raw_markdown
                        else:
                            markdown_content = str(result.markdown)
                        cache_info = "cache trống, đã crawl lại"
            else:
                markdown_content = "Không có nội dung Markdown"
            
            # Xử lý extracted_json (đảm bảo không trả về None khi dùng JSON component)
            extracted_json = result.extracted_content if hasattr(result, 'extracted_content') and result.extracted_content else "{}"
            
            # Đảm bảo extracted_json là đối tượng JSON thực sự, không phải chuỗi
            try:
                if isinstance(extracted_json, str):
                    extracted_json = json.loads(extracted_json)
                # Nếu JSON rỗng, thêm dữ liệu mẫu để kiểm tra hiển thị
                if not extracted_json:
                    extracted_json = {"info": "Không có dữ liệu JSON", "url": url}
                print(f"JSON type: {type(extracted_json)}, content: {str(extracted_json)[:100]}")
            except Exception as json_error:
                print(f"Lỗi chuyển đổi JSON: {str(json_error)}")
                # Đảm bảo luôn trả về object JSON hợp lệ
                extracted_json = {"error": "Dữ liệu JSON không hợp lệ", "url": url}
            
            # Chuyển đổi kết quả Markdown sang text để có thể sao chép
            text_content = markdown_content
            
            status = f"✅ Crawl thành công ({cache_info})! Thời gian: {duration:.2f}s"
            return status, markdown_content, extracted_json, text_content
        except Exception as e:
            duration = time.time() - start_time
            return f"❌ Lỗi: {str(e)} (sau {duration:.2f}s)", None, "{}", None
    
    def set_css_schema(self, schema):
        """Lưu schema CSS"""
        self.css_schema = schema
        return "Đã lưu schema CSS"
    
    def set_llm_config(self, provider, api_key, schema, instruction):
        """Lưu cấu hình LLM"""
        self.llm_provider = provider
        self.llm_api_key = api_key
        self.llm_schema = schema
        self.llm_instruction = instruction
        return "Đã lưu cấu hình LLM"
    
    def crawl_url_sync(self, url, cache_mode, content_threshold, extraction_mode):
        """Wrapper đồng bộ cho crawl_url"""
        global loop
        if loop is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.crawl_url(url, cache_mode, content_threshold, extraction_mode)
        )
    
    async def crawl_deep_bfs(self, url, max_depth, max_pages, include_external, cache_mode="BYPASS", content_threshold=0.1):
        """Hàm thực hiện deep crawl với BFS strategy"""
        # Tạo bộ lọc nội dung và trình tạo markdown
        content_filter = PruningContentFilter(threshold=float(content_threshold), threshold_type="fixed")
        md_generator = DefaultMarkdownGenerator(content_filter=content_filter)
        
        # Tạo BFS Deep Crawl Strategy
        bfs_strategy = BFSDeepCrawlStrategy(
            max_depth=int(max_depth),
            max_pages=int(max_pages),
            include_external=include_external
        )
        
        # Cấu hình CrawlerRunConfig
        run_config = CrawlerRunConfig(
            cache_mode=getattr(CacheMode, cache_mode),
            markdown_generator=md_generator,
            deep_crawl_strategy=bfs_strategy,
            stream=True
        )
        
        # Thực hiện crawl
        start_time = time.time()
        results_data = []
        
        # Lưu trữ dữ liệu đầy đủ của mỗi trang
        all_results = []
        
        # Theo dõi URL đã xử lý để tránh trùng lặp
        processed_urls = set()
        
        try:
            # Lấy instance crawler hiện có hoặc tạo mới
            crawler = await self.init_crawler()
            
            # Thực hiện deep crawl với streaming mode
            async for result in await crawler.arun(url=url, config=run_config):
                # Kiểm tra URL có bị trùng lặp không
                current_url = result.url
                if current_url in processed_urls:
                    continue  # Bỏ qua URL đã xử lý
                
                # Thêm URL vào danh sách đã xử lý
                processed_urls.add(current_url)
                
                # Lấy thông tin markdown
                if hasattr(result, 'markdown'):
                    if hasattr(result.markdown, 'fit_markdown'):
                        markdown_content = result.markdown.fit_markdown
                    elif hasattr(result.markdown, 'raw_markdown'):
                        markdown_content = result.markdown.raw_markdown
                    else:
                        markdown_content = str(result.markdown)
                else:
                    markdown_content = "Không có nội dung Markdown"
                
                # Lấy thông tin metadata
                depth = result.metadata.get('depth', 0) if hasattr(result, 'metadata') else 0
                
                # Xử lý extracted_json nếu có
                json_content = result.extracted_content if hasattr(result, 'extracted_content') and result.extracted_content else "{}"
                
                # Đảm bảo json_content là đối tượng JSON thực sự, không phải chuỗi
                try:
                    if isinstance(json_content, str):
                        json_content = json.loads(json_content)
                    # Nếu JSON rỗng, thêm dữ liệu mẫu có cấu trúc
                    if not json_content:
                        json_content = {
                            "url": current_url,
                            "depth": depth,
                            "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "success"
                        }
                    print(f"Deep JSON type: {type(json_content)}, content: {str(json_content)[:100]}")
                except Exception as json_error:
                    print(f"Lỗi chuyển đổi JSON deep crawl: {str(json_error)}")
                    # Đảm bảo luôn trả về object JSON hợp lệ
                    json_content = {
                        "error": "Dữ liệu JSON không hợp lệ", 
                        "url": current_url,
                        "depth": depth
                    }
                
                # Lưu thông tin đầy đủ của trang
                all_results.append({
                    "url": current_url,
                    "depth": depth,
                    "markdown": markdown_content,
                    "text": markdown_content,
                    "json": json_content
                })
                
                # Thêm kết quả vào danh sách hiển thị
                results_data.append([
                    current_url,
                    depth,
                    markdown_content[:200] + "..." if len(markdown_content) > 200 else markdown_content
                ])
            
            duration = time.time() - start_time
            status = f"✅ Deep crawl hoàn tất! Đã crawl {len(results_data)}/{max_pages} trang độc nhất. Thời gian: {duration:.2f}s"
            
            # Mặc định hiển thị nội dung của trang đầu tiên (nếu có)
            first_result = all_results[0] if all_results else {"markdown": "", "text": "", "json": {"info": "Không có dữ liệu"}}
            
            # Kiểm tra loại dữ liệu JSON trước khi trả về
            print(f"First result JSON type: {type(first_result['json'])}")
            
            return status, results_data, first_result["markdown"], first_result["text"], first_result["json"], all_results
        except Exception as e:
            duration = time.time() - start_time
            return f"❌ Lỗi: {str(e)} (sau {duration:.2f}s)", [], "", "", {"error": str(e)}, []

    def crawl_deep_bfs_sync(self, url, max_depth, max_pages, include_external, cache_mode, content_threshold):
        """Wrapper đồng bộ cho crawl_deep_bfs"""
        global loop
        if loop is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.crawl_deep_bfs(url, max_depth, max_pages, include_external, cache_mode, content_threshold)
        )
    
    def get_selected_result(self, evt: gr.SelectData, results):
        """Lấy thông tin chi tiết của kết quả được chọn từ bảng"""
        if not results or evt.index[0] >= len(results):
            return "", "", {}
        
        selected = results[evt.index[0]]
        return selected["markdown"], selected["text"], selected["json"]

# Khởi tạo ứng dụng
app = CrawlApp()

# Hướng dẫn sử dụng CSS
CSS_GUIDE = """
### Hướng dẫn cấu hình CSS Selector

CSS selector giúp bạn trích xuất dữ liệu có cấu trúc từ trang web dựa trên các thẻ HTML và class.

**Định dạng schema cơ bản:**
```json
{
    "name": "Tên nhóm dữ liệu",
    "baseSelector": "CSS selector gốc để chọn nhiều phần tử",
    "fields": [
        {
            "name": "tên_trường",
            "selector": "selector con",
            "type": "text|attribute|html|innerHtml", 
            "attribute": "tên_thuộc_tính" (chỉ cần khi type là attribute)
        }
    ]
}
```

**Ví dụ thực tế - Trích xuất sản phẩm từ trang thương mại điện tử:**
```json
{
    "name": "Products",
    "baseSelector": ".product-item",
    "fields": [
        {"name": "title", "selector": "h3.product-title", "type": "text"},
        {"name": "price", "selector": ".price-current", "type": "text"},
        {"name": "image", "selector": "img.product-image", "type": "attribute", "attribute": "src"},
        {"name": "url", "selector": "a.product-link", "type": "attribute", "attribute": "href"}
    ]
}
```

**Mẹo sử dụng:**
1. Sử dụng DevTools trong trình duyệt (F12) để xác định CSS selectors
2. Selector `.classname` chọn phần tử theo class
3. Selector `#id` chọn phần tử theo ID
4. Selector `element` chọn tất cả thẻ HTML đó (vd: `a`, `div`, `img`)
5. Các types: 
   - `text`: Lấy nội dung văn bản
   - `attribute`: Lấy giá trị thuộc tính (href, src, alt, v.v)
   - `html`: Lấy HTML đầy đủ bao gồm thẻ
   - `innerHtml`: Lấy HTML bên trong thẻ

Sau khi lưu schema, chọn chế độ trích xuất "css" ở tab Crawl cơ bản để sử dụng.
"""

# Hướng dẫn sử dụng LLM
LLM_GUIDE = """
### Hướng dẫn cấu hình LLM (Language Model)

LLM giúp bạn trích xuất dữ liệu có cấu trúc từ trang web bằng cách sử dụng mô hình ngôn ngữ lớn.

**Các thành phần cấu hình:**

1. **LLM Provider**: Chọn nhà cung cấp và mô hình
   - **Ollama** (mô hình cục bộ, miễn phí): cần cài đặt Ollama trên máy tính
   - **OpenAI** (API có phí): yêu cầu API key

2. **API Key**: Cần thiết cho OpenAI. Không bắt buộc cho Ollama.

3. **Schema LLM**: Cấu trúc dữ liệu bạn muốn trích xuất, ở dạng JSON với kiểu dữ liệu.

4. **Hướng dẫn trích xuất**: Chỉ dẫn cụ thể cho LLM về cách trích xuất dữ liệu.

**Ví dụ schema cho trang sản phẩm:**
```json
{
    "title": "string",
    "price": "string",
    "description": "string", 
    "rating": "string",
    "specifications": {
        "brand": "string",
        "model": "string",
        "size": "string",
        "weight": "string"
    }
}
```

**Ví dụ hướng dẫn trích xuất:**
```
Từ trang sản phẩm, hãy trích xuất:
1. Tiêu đề chính của sản phẩm
2. Giá hiện tại (bỏ qua giá gốc)
3. Mô tả sản phẩm (đoạn văn đầu tiên)
4. Đánh giá (số sao hoặc điểm)
5. Thông số kỹ thuật bao gồm thương hiệu, mẫu, kích thước, và trọng lượng nếu có
```

**Lưu ý quan trọng:**
- Khi sử dụng OpenAI, đảm bảo API key của bạn còn hạn dùng
- Mô hình mới nhất thường cho kết quả tốt nhất (GPT-4.1, GPT-4o)
- Ollama yêu cầu cài đặt và chạy trên máy tính cục bộ
- Trích xuất bằng LLM chậm hơn CSS nhưng thông minh hơn, phù hợp với dữ liệu phức tạp

Sau khi lưu cấu hình, chọn chế độ trích xuất "llm" ở tab Crawl cơ bản để sử dụng.
"""

# Xây dựng giao diện Gradio
with gr.Blocks(title="Crawl4AI UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕸️ Crawl4AI - Giao diện người dùng")
    
    with gr.Tabs():
        with gr.TabItem("Crawl cơ bản"):
            with gr.Row():
                with gr.Column():
                    url_input = gr.Textbox(label="URL", placeholder="https://example.com")
                    with gr.Row():
                        cache_mode = gr.Dropdown(
                            choices=["BYPASS", "ENABLED"], 
                            value="BYPASS", 
                            label="Chế độ cache"
                        )
                        content_threshold = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.1, step=0.1,
                            label="Ngưỡng nội dung"
                        )
                    extraction_mode = gr.Radio(
                        choices=["markdown", "css", "llm"], 
                        value="markdown",
                        label="Chế độ trích xuất"
                    )
                    crawl_btn = gr.Button("Bắt đầu Crawl", variant="primary")
                
                with gr.Column():
                    status_output = gr.Textbox(label="Trạng thái")
                    with gr.Tabs():
                        with gr.TabItem("Markdown"):
                            markdown_output = gr.Markdown(label="Kết quả Markdown")
                        with gr.TabItem("Text"):
                            text_output = gr.Textbox(
                                label="Kết quả Văn bản",
                                lines=20,
                                max_lines=50,
                                show_copy_button=True
                            )
                        with gr.TabItem("JSON"):
                            json_output = gr.JSON(label="Kết quả JSON")
        
        with gr.TabItem("Deep Crawl (BFS)"):
            with gr.Row():
                with gr.Column():
                    deep_url_input = gr.Textbox(label="URL", placeholder="https://example.com")
                    with gr.Row():
                        max_depth = gr.Slider(
                            minimum=1, maximum=3, value=1, step=1,
                            label="Độ sâu tối đa"
                        )
                        max_pages = gr.Slider(
                            minimum=5, maximum=100, value=20, step=5,
                            label="Số trang tối đa"
                        )
                    with gr.Row():
                        include_external = gr.Checkbox(
                            value=False,
                            label="Bao gồm link ngoài"
                        )
                        deep_cache_mode = gr.Dropdown(
                            choices=["BYPASS", "ENABLED"], 
                            value="BYPASS", 
                            label="Chế độ cache"
                        )
                        deep_content_threshold = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.1, step=0.1,
                            label="Ngưỡng nội dung"
                        )
                    deep_crawl_btn = gr.Button("Bắt đầu Deep Crawl", variant="primary")
                
                with gr.Column():
                    deep_status_output = gr.Textbox(label="Trạng thái")
                    with gr.Tabs():
                        with gr.TabItem("Kết quả"):
                            deep_results = gr.Dataframe(
                                headers=["URL", "Độ sâu", "Nội dung tóm tắt"],
                                label="Kết quả Deep Crawl (Click vào một dòng để xem chi tiết)",
                                wrap=True,
                                interactive=False
                            )
                        with gr.TabItem("Markdown"):
                            deep_markdown_output = gr.Markdown(label="Nội dung Markdown")
                        with gr.TabItem("Text"):
                            deep_text_output = gr.Textbox(
                                label="Nội dung văn bản",
                                lines=20,
                                max_lines=50,
                                show_copy_button=True
                            )
                        with gr.TabItem("JSON"):
                            deep_json_output = gr.JSON(label="Kết quả JSON", visible=True)
                    
                    # Biến ẩn để lưu trữ tất cả kết quả
                    all_deep_results = gr.State([])
        
        with gr.TabItem("Cấu hình CSS"):
            
            with gr.Accordion("Nhập Schema CSS", open=True):
                css_schema = gr.Textbox(
                    label="Schema CSS (JSON)", 
                    placeholder='''{
    "name": "Example Items",
    "baseSelector": "div.item",
    "fields": [
        {"name": "title", "selector": "h2", "type": "text"},
        {"name": "link", "selector": "a", "type": "attribute", "attribute": "href"}
    ]
}''',
                    lines=10
                )
                save_css_btn = gr.Button("Lưu schema CSS")
                css_status = gr.Textbox(label="Trạng thái")

            gr.Markdown(CSS_GUIDE)
        
        with gr.TabItem("Cấu hình LLM"):
            
            with gr.Accordion("Cấu hình LLM", open=True):
                with gr.Row():
                    llm_provider = gr.Dropdown(
                        choices=["ollama/llama3.3", "openai/gpt-4.1", "openai/gpt-4.1-mini", "openai/gpt-4.1-nano", "openai/gpt-4o", "openai/gpt-4o-mini"], 
                        value="ollama/llama3.3",
                        label="LLM Provider"
                    )
                    llm_api_key = gr.Textbox(
                        label="API Key (cần thiết cho OpenAI)", 
                        placeholder="sk-...",
                        type="password"
                    )
                
                llm_schema = gr.Textbox(
                    label="Schema LLM (JSON)",
                    placeholder='''{
    "title": "string",
    "description": "string",
    "price": "string",
    "rating": "string"
}''',
                    lines=8
                )
                
                llm_instruction = gr.Textbox(
                    label="Hướng dẫn trích xuất",
                    placeholder="Trích xuất tiêu đề, mô tả, giá và đánh giá từ trang sản phẩm này.",
                    lines=3
                )
                
                save_llm_btn = gr.Button("Lưu cấu hình LLM")
                llm_status = gr.Textbox(label="Trạng thái")

            gr.Markdown(LLM_GUIDE)
    
    # Đăng ký các hàm xử lý sự kiện
    crawl_btn.click(
        app.crawl_url_sync,
        inputs=[url_input, cache_mode, content_threshold, extraction_mode],
        outputs=[status_output, markdown_output, json_output, text_output]
    )
    
    deep_crawl_btn.click(
        app.crawl_deep_bfs_sync,
        inputs=[deep_url_input, max_depth, max_pages, include_external, deep_cache_mode, deep_content_threshold],
        outputs=[deep_status_output, deep_results, deep_markdown_output, deep_text_output, deep_json_output, all_deep_results]
    )
    
    # Sự kiện khi người dùng bấm vào một dòng trong bảng kết quả
    deep_results.select(
        app.get_selected_result,
        inputs=[all_deep_results],
        outputs=[deep_markdown_output, deep_text_output, deep_json_output]
    )
    
    save_css_btn.click(
        app.set_css_schema,
        inputs=[css_schema],
        outputs=[css_status]
    )
    
    save_llm_btn.click(
        app.set_llm_config,
        inputs=[llm_provider, llm_api_key, llm_schema, llm_instruction],
        outputs=[llm_status]
    )

# Khởi chạy giao diện
if __name__ == "__main__":
    # Khởi tạo vòng lặp sự kiện
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Kết thúc crawler khi ứng dụng đóng
    try:
        demo.launch()
    finally:
        if crawler_instance is not None:
            loop.run_until_complete(crawler_instance.stop())
        loop.close() 